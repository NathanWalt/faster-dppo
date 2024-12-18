import torch
from torch import nn
import torch.nn.functional as F

# a w8a8 linear layer
# activation per-tensor, weight per-channel
class QLinear(nn.Module):
    def __init__(self, org_linear: nn.Linear, name: str):
        super().__init__()
        self.org_linear = org_linear
        self.name = name
        
        w = org_linear.weight
        weight_scale = torch.max(torch.abs(w)) / 127
        weight_int8 = torch.round(w / weight_scale).to(torch.int8)
        self.register_buffer('weight_int8', weight_int8)
        self.register_buffer('weight_scale', weight_scale)
        self.register_buffer('bias', org_linear.bias)
        
        del org_linear
    
    def forward(self, x):
        x_scales = (torch.max(torch.abs(x), dim=-1, keepdim=True)[0] / 127) + 1e-8
        x = (x / x_scales).round() * x_scales
        w = self.weight_int8.to(x.dtype) * self.weight_scale
        y = F.linear(x, w, self.bias)
        return y

def quant_model(model: nn.Module, omitted_layers_blocks:list=[]):
    assert isinstance( model, nn.Module)
    for name, child_module in model.named_children():
        if name in omitted_layers_blocks:
            print(f'Omitting layer: {name}')
            continue
        if isinstance(child_module, nn.Linear):
            setattr(model, name, QLinear(child_module, name))
            print(f'quantize Linear: {name}')
        else:
            quant_model(child_module)
    return model