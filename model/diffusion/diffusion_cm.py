import logging
import torch
from torch import nn
import torch.nn.functional as F
from model.diffusion.diffusion import Sample
from model.diffusion.diffusion_ct import CTDiffusionModel

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
        sampling_steps = 1000,
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
        log.info("CTConsistencyModel with time steps %d", sampling_steps)
        self.horizon_steps = horizon_steps

        # Set up models
        self.network = network.to(device)
        if network_path is not None:
            checkpoint = torch.load(
                network_path, map_location=device, weights_only=True
            )
            if "ema" in checkpoint:
                self.load_state_dict(checkpoint["ema"], strict=False)
                logging.info("Loaded SL-trained policy from %s", network_path)
            else:
                self.load_state_dict(checkpoint["model"], strict=False)
                logging.info("Loaded RL-trained policy from %s", network_path)
        logging.info("initialized CTConsistencyModel")
        logging.info(
            f"Number of network parameters: {sum(p.numel() for p in self.parameters())}"
        )

    def get_c(self,t):
        sigma_data = torch.tensor(0.5)
        a = (sigma_data * sigma_data)/(t * t + sigma_data * sigma_data)
        b = sigma_data * t / torch.sqrt(t * t + sigma_data * sigma_data)
        return a, b

    def forward(self, cond, deterministic=True, t = None, x = None):
        device = self.device
        B = len(cond["state"])

        if t is None:
            t = torch.ones((B, 1, 1), device=device)

        if x is None:
            x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)

        # a, b = self.get_c(t)
        # x = a * x + b * self.network(x, t, cond)
        x = self.network(x, t, cond) # when t=1, x is the output of the network

        return Sample(x, None)

    def loss(self, teacher_model, ema_model, x, *args):
        return self.consistency_loss(ema_model, teacher_model, x, *args)

    def get_t_i(self, i, sampling_steps):
        return i / sampling_steps


    def consistency_loss(self, ema_model, teacher_model : CTDiffusionModel, x, cond: dict):
        batch_size = x.shape[0]
        n = torch.randint(1, self.sampling_steps, (batch_size, 1, 1)).cuda()

        tn1 = self.get_t_i(n+1, self.sampling_steps)
        x_tn1 = teacher_model.q_sample(x, tn1, None)
        x_tn_phi = x_tn1 + (self.get_t_i(n, self.sampling_steps) - self.get_t_i(n+1, self.sampling_steps)) * teacher_model.pfode_dxt(x_tn1, torch.tensor(self.get_t_i(n+1, self.sampling_steps)), cond)

        # a, b = self.get_c(self.get_t_i(n+1, sampling_steps))

        max_std = teacher_model.get_std(torch.tensor(1.0))
        a = teacher_model.get_mu_coeff(self.get_t_i(n+1, self.sampling_steps))
        b = teacher_model.get_std(self.get_t_i(n+1, self.sampling_steps)) / max_std
        f_theta = a * x_tn1 + b * self.network(x_tn1, self.get_t_i(n+1, self.sampling_steps), cond=cond)

        a = teacher_model.get_mu_coeff(self.get_t_i(n, self.sampling_steps))
        b = teacher_model.get_std(self.get_t_i(n, self.sampling_steps)) /max_std
        with torch.no_grad():
            f_theta_minus = a * x_tn_phi + b * ema_model.network(x_tn_phi, self.get_t_i(n, self.sampling_steps), cond=cond)

        loss = F.mse_loss(f_theta, f_theta_minus, reduction="mean")
        return self.consistency_lambda * loss