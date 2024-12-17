import logging
import torch
from torch import nn
import torch.nn.functional as F

from model.diffusion.diffusion import Sample

log = logging.getLogger(__name__)


class CTDiffusionModel(nn.Module):

    def __init__(
        self,
        network,
        horizon_steps,
        obs_dim,
        action_dim,
        network_path=None,
        device="cuda:0",
        # Various clipping
        denoised_clip_value=1.0,
        randn_clip_value=10,
        final_action_clip_value=None,
        eps_clip_value=None,  # DDIM only
        # DDPM parameters
        beta_min=0.1,
        beta_max=20.0,
        predict_epsilon=True,
        sampling_steps=100,
        # DDIM sampling
        use_ddim=False,
        ddim_discretize="uniform",
        ddim_steps=None,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.horizon_steps = horizon_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.sampling_steps = sampling_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.predict_epsilon = predict_epsilon
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps

        # Clip noise value at each denoising step
        self.denoised_clip_value = denoised_clip_value

        # Whether to clamp the final sampled action between [-1, 1]
        self.final_action_clip_value = final_action_clip_value

        # For each denoising step, we clip sampled randn (from standard deviation) such that the sampled action is not too far away from mean
        self.randn_clip_value = randn_clip_value

        # Clip epsilon for numerical stability
        self.eps_clip_value = eps_clip_value

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
        logging.info(
            f"Number of network parameters: {sum(p.numel() for p in self.parameters())}"
        )

    def get_beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def beta_integral(self, s, t):
        res = self.beta_min + 0.5*(self.beta_max - self.beta_min)*(t + s)
        res *= (t - s)
        return res

    def get_mu_coeff(self, t):
        return torch.exp(-0.5 * self.beta_integral(0, t))

    def get_std(self, t):
        return torch.sqrt(1 - torch.exp(-self.beta_integral(0, t)))

    # ---------- Sampling ----------#

    def pfode_dxt(self, x, t, cond):
        noise_prediction = self.network(x, t, cond=cond)
        sigma_t = self.get_std(t)
        if self.predict_epsilon:
            score = - noise_prediction / sigma_t
        else:
            score = (x - noise_prediction) / (sigma_t ** 2)
        beta_t = self.get_beta(t)
        dxt = -0.5 * beta_t * (x + score)
        return dxt

    @torch.no_grad()
    def forward(self, cond, deterministic=True, n_timesteps=None):
        device = self.device
        B = len(cond["state"])

        if n_timesteps is None:
            n_timesteps = self.sampling_steps

        # a naive implementation euler-solver of PF-ODE
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        t_all = torch.arange(n_timesteps, 0, -1, device=device) / n_timesteps
        dt = 1.0 / n_timesteps
        for i, t in enumerate(t_all):
            t = t.view(-1, 1, 1)
            t = t.expand(x.shape[0], 1, 1)
            dxt = self.pfode_dxt(x, t, cond)
            x = x - dxt * dt

            # clamp action at final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(
                    x, -self.final_action_clip_value, self.final_action_clip_value
                )
        return Sample(x, None)


    # ---------- Supervised training ----------#

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.rand(
            (batch_size, 1, 1),
            device=x.device
        )
        return self.p_losses(x, *args, t=t)

    def p_losses(self, x_start, cond: dict, t):
        device = x_start.device

        # Forward process
        noise = torch.randn_like(x_start, device=device)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Predict
        x_recon = self.network(x_noisy, t, cond=cond)
        if self.predict_epsilon:
            loss = F.mse_loss(x_recon, noise, reduction="mean")
        else:
            loss = F.mse_loss(x_recon, x_start, reduction="mean")
        return loss

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)
        return (
                self.get_mu_coeff(t) * x_start
                + self.get_std(t) * noise
        )
