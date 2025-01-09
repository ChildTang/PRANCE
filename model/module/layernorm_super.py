import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LayerNormSuper(torch.nn.LayerNorm):
    def __init__(self, super_embed_dim):
        super().__init__(super_embed_dim)

        # the largest embed dim
        self.super_embed_dim = super_embed_dim

        # the current sampled embed dim
        self.sample_embed_dim = None

        self.samples = {}
        self.profiling = False

    def forward(self, x, sample_embed_dim, token_mask, batch_inference=False):
        B, N, C = x.shape

        self.sample_embed_dim = sample_embed_dim
        weight = self.weight
        bias = self.bias

        if batch_inference:
            # Perform LayerNorm directly without using mask
            x = F.layer_norm(x, (C,), eps=self.eps)

            # Adjust weight and bias for the current embed dim
            weight = self.weight[:C]
            bias = self.bias[:C]

            # Calculate the scaling ratio
            zoom_ratio = torch.sqrt(
                torch.tensor(sample_embed_dim) / self.super_embed_dim
            ).to(x.device)
            x = x * zoom_ratio.unsqueeze(-1).unsqueeze(-1) * weight + bias

        else:
            mask = torch.arange(C).unsqueeze(0).unsqueeze(0) < torch.tensor(
                sample_embed_dim
            ).unsqueeze(1).unsqueeze(2)
            mask = mask.to("cuda")

            mask_x = x * mask
            sum_values = torch.sum(mask_x, dim=-1)
            num_valid_elements = torch.sum(mask, dim=-1)
            mean = sum_values / num_valid_elements

            # Fill the clipped parts with the mean
            mean = mean.unsqueeze(2).expand_as(x)
            filled_mean = mean * (1 - mask.float())

            # Fill the clipped parts with the mean
            x = mask_x + filled_mean

            # Layernorm
            x = F.layer_norm(x, (x.shape[-1],), eps=self.eps)

            # Calculate the scaling ratio
            zoom_ratio = torch.sqrt(torch.tensor(sample_embed_dim) / C).to("cuda")
            x = x * zoom_ratio.unsqueeze(-1).unsqueeze(-1) * weight + bias

            x *= mask

            if token_mask is not None:
                x *= token_mask

        return x

    def calc_sampled_param_num(self):
        super_param_num = self.weight.numel() + self.bias.numel()
        zoom_ratio = self.sample_embed_dim / self.super_embed_dim
        param_num = np.array(super_param_num * zoom_ratio)

        return param_num

    def get_complexity(self, sequence_length):
        return np.array(sequence_length * self.sample_embed_dim).astype(np.float64)
