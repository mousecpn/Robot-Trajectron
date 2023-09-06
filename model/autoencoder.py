import torch
from torch.nn import Module
import torch.nn as nn
import model.diffusion as diffusion
from model.diffusion import DiffusionTraj,VarianceSchedule
import pdb

class AutoEncoder(Module):

    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.diffnet = getattr(diffusion, config.diffnet)

        self.diffusion = DiffusionTraj(
            net = self.diffnet(point_dim=3, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False),
            var_sched = VarianceSchedule(
                num_steps=100,
                beta_T=5e-2,
                mode='linear'
            )
        )

    def encode(self, batch):
        z = self.encoder.get_latent(batch)
        return z
    
    def generate(self, batch, num_points, sample, bestof,flexibility=0.0, ret_traj=False, sampling="ddpm", step=100):
        #print(f"Using {sampling}")
        dynamics = self.encoder.model.dynamic
        encoded_x = self.encoder.get_latent(batch)
        predicted_y_vel =  self.diffusion.sample(num_points, encoded_x,sample,bestof, flexibility=flexibility, ret_traj=ret_traj, sampling=sampling, step=step)
        predicted_y_pos = dynamics.integrate_samples(predicted_y_vel)
        return predicted_y_pos.cpu().detach().numpy()

    def get_loss(self, batch):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t) = batch

        feat_x_encoded = self.encode(batch) # B * 64
        loss = self.diffusion.get_loss(y_t.cuda(), feat_x_encoded)
        return loss
