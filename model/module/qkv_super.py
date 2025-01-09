import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class qkv_super(nn.Linear):
    def __init__(
        self,
        super_in_dim,
        super_out_dim,
        bias=True,
        uniform_=None,
        non_linear="linear",
        scale=False,
    ):
        super().__init__(super_in_dim, super_out_dim, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples = {}

        self.scale = scale
        # self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False

    def _reset_parameters(self, bias, uniform_, non_linear):
        (
            nn.init.xavier_uniform_(self.weight)
            if uniform_ is None
            else uniform_(self.weight, non_linear=non_linear)
        )
        if bias:
            nn.init.constant_(self.bias, 0.0)

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim

        self._sample_parameters()

    def _sample_parameters(self):
        # Since x already has a mask, use the entire w_in here. Note that self.sample_out_dim must be used here.
        self.samples["weight"] = sample_weight(
            self.weight, self.super_in_dim, self.sample_out_dim
        )
        self.samples["bias"] = self.bias.clone() if self.bias is not None else None

        self.sample_scale = self.super_out_dim / self.sample_out_dim
        if self.bias is not None:
            self.samples["bias"] = sample_bias(self.bias, self.sample_out_dim)

        return self.samples

    def forward(
        self, x, sample_in_dim, sample_out_dim, token_length=None, batch_inference=False
    ):
        # The sample_out_dim here is only related to the number of heads, but this dimension is fixed, so it will not change
        self.set_sample_config(sample_in_dim, sample_out_dim)
        B, N, C = x.shape

        if batch_inference:
            # Perform Linear directly without using mask
            self.samples["weight"] = sample_weight(self.weight, C, self.sample_out_dim)
            x = F.linear(x, self.samples["weight"], self.samples["bias"]) * (
                self.sample_scale if self.scale else 1
            )

        else:
            x = F.linear(x, self.samples["weight"], self.samples["bias"]) * (
                self.sample_scale if self.scale else 1
            )

            ##################################################################################
            # Add mask for tokens
            # First term: [1, N, 1]
            # Second term: [B, 1, 1]
            if token_length is not None:
                B, N, C = x.shape
                mask_indices = torch.arange(N).unsqueeze(0).unsqueeze(
                    -1
                ) < torch.tensor(token_length).unsqueeze(1).unsqueeze(2)
                token_mask = mask_indices.expand(B, N, C).float().to("cuda")

                x *= token_mask
            ##################################################################################

        return x

    def calc_sampled_param_num(self):
        zoom_in_ratio = self.sample_in_dim / self.super_in_dim
        zoom_out_ratio = self.sample_out_dim / self.super_out_dim
        weight_numel = np.array(self.weight.numel() * zoom_in_ratio * zoom_out_ratio)

        if self.samples["bias"] is not None:
            bias_numel = np.array(self.bias.numel() * zoom_out_ratio)
        else:
            bias_numel = 0

        return weight_numel + bias_numel

    def get_complexity(self, sequence_length):
        total_flops = 0

        zoom_in_ratio = self.sample_in_dim / self.super_in_dim
        zoom_out_ratio = self.sample_out_dim / self.super_out_dim

        total_flops += sequence_length * np.prod(self.weight.size())

        final_flops = total_flops * zoom_in_ratio * zoom_out_ratio

        return np.array(final_flops).astype(np.float64)


def sample_weight(weight, sample_in_dim, sample_out_dim):
    sample_weight = weight[:, :sample_in_dim]
    sample_weight = torch.cat(
        [sample_weight[i:sample_out_dim:3, :] for i in range(3)], dim=0
    )

    return sample_weight


def sample_bias(bias, sample_out_dim):
    sample_bias = bias[:sample_out_dim]

    return sample_bias
