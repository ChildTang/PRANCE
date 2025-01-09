import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import to_2tuple
import numpy as np


class PatchembedSuper(nn.Module):
    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, scale=False
    ):
        super(PatchembedSuper, self).__init__()

        # parameters
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # embedding
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.super_embed_dim = embed_dim
        self.scale = scale

        # sampled_
        self.sample_embed_dim = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_scale = None

    def forward(self, x, sample_embed_dim, batch_inference=False):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        self.sample_embed_dim = sample_embed_dim

        if self.scale:
            self.sampled_scale = self.super_embed_dim / sample_embed_dim

        x = (
            F.conv2d(
                x,
                self.proj.weight,
                self.proj.bias,
                stride=self.patch_size,
                padding=self.proj.padding,
                dilation=self.proj.dilation,
            )
            .flatten(2)
            .transpose(1, 2)
        )

        # # pruning
        if batch_inference:
            x = x[:, :, : self.sample_embed_dim[0]]

        else:
            # create mask
            mask = torch.ones_like(x)

            # Convert sample_embed_dim to a tensor, [B, 1, 1]
            sample_embed_dim_tensor = (
                torch.tensor(sample_embed_dim).unsqueeze(1).unsqueeze(2)
            )

            # Update mask according to sample_embed_dim
            # The shape of torch.arange(x.shape[2]).unsqueeze(0).unsqueeze(0) is [1, 1, C]
            # The shape of mask is [B, 1, C]
            mask = (
                torch.arange(x.shape[2]).unsqueeze(0).unsqueeze(0)
                < sample_embed_dim_tensor
            )

            # Element-wise multiplication using the pruning mask
            x *= mask.to("cuda")

        if self.scale:
            return x * self.sampled_scale
        return x

    def calc_sampled_param_num(self):
        super_param_num = self.proj.weight.numel() + self.proj.bias.numel()
        zoom_ration = self.sample_embed_dim / self.super_embed_dim
        param_num = np.array(super_param_num * zoom_ration)

        return param_num

    def get_complexity(self, sequence_length):
        total_flops = 0

        zoom_ratio = self.sample_embed_dim / self.super_embed_dim

        total_flops += self.proj.bias.size(0)
        total_flops += sequence_length * np.prod(self.proj.weight.size())

        final_flops = np.array(total_flops * zoom_ratio).astype(np.float64)

        return final_flops
