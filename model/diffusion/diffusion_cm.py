import logging
import torch
from torch import nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class CTConsistencyModel(nn.Module):

    def __init__(
        self,
        network,
        obs_dim,
        action_dim,
        horizon_steps,
        network_path=None,
        device="cuda:0",
        sampling_steps = 100,
        # Consistency parameters
        consistency_lambda=1.0,  # Consistency loss weight
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.consistency_lambda = consistency_lambda
        self.sampling_steps = sampling_steps
        self.horizon_steps = horizon_steps
        
        # Set up models
        self.network = network.to(device)
        # if network_path is not None:
        #     checkpoint = torch.load(
        #         network_path, map_location=device, weights_only=True
        #     )
        #     self.load_state_dict(checkpoint["model"], strict=False)
        #     logging.info("Loaded model from %s", network_path)
        logging.info("initialized CTConsistencyModel")
        logging.info(
            f"Number of network parameters: {sum(p.numel() for p in self.parameters())}"
        )
        
    def get_c(self,t):
        sigma_data = torch.tensor(0.5)
        a = (sigma_data * sigma_data)/(t * t + sigma_data * sigma_data)
        b = sigma_data * t / torch.sqrt(t * t + sigma_data * sigma_data)
        return a, b

    def forward(self, cond, deterministic=True, t = 1.0, x = None):
        device = self.device
        B = len(cond["state"])
        
        if x is None:
            x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        
        a, b = self.get_c(t)
        x = a * x + b * self.network(x, t, cond)
        
        return x

    def loss(self, teacher_model, ema_model, sampling_steps, x, *args):
        return self.consistency_loss(ema_model, teacher_model, sampling_steps, x, *args)
    
    def get_t_i(self, i):
        return torch.tensor(1.0) / i
    
    def consistency_loss(self, ema_model, teacher_model, sampling_steps, x, cond: dict):
        batch_size = x.shape[0]
        n = torch.randint(1, sampling_steps, (batch_size, 1, 1)).cuda()
        
        x_tn1 = x + torch.randn_like(x) * torch.sqrt(self.get_t_i(n+1)**2)
        x_tn_phi = x_tn1 + (self.get_t_i(n) - self.get_t_i(n+1)) * teacher_model.pfode_dxt(x_tn1, torch.tensor(self.get_t_i(n+1)), cond)
        
        a, b = self.get_c(self.get_t_i(n+1))
        f_theta = a * x + b * self.network(x_tn1, self.get_t_i(n+1), cond=cond)
        
        a, b = self.get_c(self.get_t_i(n))
        f_theta_minus = a * x_tn_phi + b * ema_model.network(x_tn_phi, self.get_t_i(n), cond=cond)

        loss = F.mse_loss(f_theta, f_theta_minus, reduction="mean")
        return self.consistency_lambda * loss