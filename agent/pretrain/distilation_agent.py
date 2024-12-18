import torch
from torch import nn, optim
from agent.pretrain.train_agent import PreTrainAgent, batch_to_device
from model.diffusion.diffusion_ct import CTDiffusionModel
from model.diffusion.diffusion_cm import CTConsistencyModel
from util.timer import Timer
import logging
log = logging.getLogger(__name__)
import wandb
import hydra

import numpy as np
class ConsistencyDistillationAgent(PreTrainAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.teacher_model : CTDiffusionModel = hydra.utils.instantiate(cfg.teacher_model)
        self.update_ema_freq = 1
        self.epoch_start_ema = 0

        loadpath = cfg.teacher_model_network_path
        device = "cuda:0"
        data = torch.load(loadpath, weights_only=True, map_location=device)
        self.teacher_model.load_state_dict(data["model"])
        self.teacher_model.requires_grad_(False)
        self.ema_model.requires_grad_(False)

        self.learning_rate = cfg.train.learning_rate
        self.d = nn.MSELoss()
        self.lambda_ = 1.0
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    # --------------------- RUN -------------------------#
    def run(self):

        timer = Timer()
        self.epoch = 1
        for _ in range(self.n_epochs):

            # train
            loss_train_epoch = []
            for batch_train in self.dataloader_train:
                if self.dataset_train.device == "cpu":
                    batch_train = batch_to_device(batch_train)

                self.model.train()

                loss_train = self.model.loss(self.teacher_model, self.ema_model, *batch_train)
                loss_train.backward()
                loss_train_epoch.append(loss_train.item())

                self.optimizer.step()
                self.optimizer.zero_grad()

                # update ema

                self.step_ema()
            loss_train = np.mean(loss_train_epoch)

            # validate
            loss_val_epoch = []
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                self.model.eval()
                for batch_val in self.dataloader_val:
                    if self.dataset_val.device == "cpu":
                        batch_val = batch_to_device(batch_val)
                    loss_val, infos_val = self.model.loss(*batch_val)
                    loss_val_epoch.append(loss_val.item())
                self.model.train()
            loss_val = np.mean(loss_val_epoch) if len(loss_val_epoch) > 0 else None

            # update lr
            self.lr_scheduler.step()

            # save model
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model()

            # log loss
            if self.epoch % self.log_freq == 0:
                log.info(
                    f"{self.epoch}: train loss {loss_train:8.4f} | t:{timer():8.4f}"
                )
                if self.use_wandb:
                    if loss_val is not None:
                        wandb.log(
                            {"loss - val": loss_val}, step=self.epoch, commit=False
                        )
                    wandb.log(
                        {
                            "loss - train": loss_train,
                        },
                        step=self.epoch,
                        commit=True,
                    )

            # count
            self.epoch += 1