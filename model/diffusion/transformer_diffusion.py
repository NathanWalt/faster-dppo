
import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TransformerBlock(nn.Module):
    def __init__(
        self, 
        input_dim, 
        num_heads=8, 
        dropout=0.2,  # 增加 dropout 
        activation_type="ReLU",
        use_layernorm=True
    ):
        super().__init__()
        
        
        activation_dict = {
            "Mish": nn.Mish(),
            "ReLU": nn.ReLU(),
            "GELU": nn.GELU(),
            "LeakyReLU": nn.LeakyReLU()
        }
        activation = activation_dict.get(activation_type, nn.ReLU())
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(input_dim) if use_layernorm else nn.Identity()
        self.norm2 = nn.LayerNorm(input_dim) if use_layernorm else nn.Identity()
        
        # Multi-head attention with more robust configuration
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=input_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True,
            kdim=input_dim,
            vdim=input_dim
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            activation,
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim),
            nn.Dropout(dropout)  
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        residual = x
        x = self.norm1(x)
        
        # 多头注意力
        attn_output, _ = self.multihead_attn(x, x, x)
        x = residual + self.dropout(attn_output)
        x = self.norm1(x)
        # Feed-forward
        residual = x
        x = self.norm2(x)
        ffn_output = self.ffn(x)
        x = residual + self.dropout(ffn_output)
        x = self.norm2(x)
        return x

class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        action_dim,
        horizon_steps,
        cond_dim,
        time_dim=16,
        transformer_dims=[256, 256],
        num_transformer_layers=3,  
        num_heads=8,
        cond_mlp_dims=None,
        activation_type="ReLU",  
        dropout=0.2,  
        use_layernorm=True,
        use_residual_connection= False
    ):
        super().__init__()
        output_dim = action_dim * horizon_steps
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.ReLU(),  # 改为 ReLU
            nn.Linear(time_dim * 2, time_dim),
        )
        
        # Conditional MLP (optional)
        if cond_mlp_dims is not None:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, cond_mlp_dims[0]),
                nn.ReLU(),  
                *[
                    nn.Sequential(
                        nn.Linear(cond_mlp_dims[i], cond_mlp_dims[i+1]),
                        nn.ReLU()  
                    ) for i in range(len(cond_mlp_dims)-1)
                ]
            )
            input_dim = time_dim + action_dim * horizon_steps + cond_mlp_dims[-1]
        else:
            input_dim = time_dim + action_dim * horizon_steps + cond_dim
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                input_dim, 
                num_heads=num_heads, 
                dropout=dropout, 
                activation_type=activation_type,
                use_layernorm=use_layernorm
            ) for _ in range(num_transformer_layers)
        ])
        
        # Output projection with more regularization
        output_layers = []
        prev_dim = input_dim
        for dim in transformer_dims:
            output_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        output_layers.append(nn.Linear(prev_dim, output_dim))
        
        self.output_projection = nn.Sequential(*output_layers)
        
        self.time_dim = time_dim
        self.use_residual_connection = use_residual_connection

    def forward(
        self,
        x,
        time,
        cond,
        **kwargs,
    ):
        """
        x: (B, Ta, Da)
        time: (B,) or int, diffusion step
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """
        B, Ta, Da = x.shape

        # flatten chunk
        x = x.view(B, -1)

        # flatten history
        state = cond["state"].view(B, -1)

        # obs encoder
        if hasattr(self, "cond_mlp"):
            state = self.cond_mlp(state)

        # append time and cond
        time = time.view(B, 1)
        time_emb = self.time_embedding(time).view(B, self.time_dim)
        x = torch.cat([x, time_emb, state], dim=-1)
        
        # Add sequence dimension for transformer
        x_orig = x
        x = x.unsqueeze(1)
        
        # Transformer layers with optional residual connection
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        
        # Output projection
        x = x.squeeze(1)
        
       # Optional global residual connection
        if self.use_residual_connection:
            x = x + x_orig 
        
        out = self.output_projection(x)
        
        return out.view(B, Ta, Da)

