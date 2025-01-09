import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module.Linear_super import LinearSuper
from model.module.layernorm_super import LayerNormSuper
from model.module.multihead_super import AttentionSuper
from model.module.embedding_super import PatchembedSuper
from model.utils import trunc_normal_
from model.utils import DropPath
import numpy as np
import copy
import random
from .utils import (
    _map_token_mode, 
    _check_is_token_merging, 
    _check_is_token_pruning, 
    _check_is_token_pruning_then_merging
    )
from .ppo_sbs import PPO

def sample_configs(choices, batch_size, sample_max=False):
    sample_max = False

    config = {}
    dimensions = ["mlp_ratio"]
    depth = choices["depth"][0]

    for dimension in dimensions:
        config[dimension] = [
            (
                random.choice(choices[dimension])
                if not sample_max
                else max(choices[dimension])
            )
            for _ in range(depth)
        ]

    embed_dim = []
    group = depth // 3
    for i in range(group):
        embed_dim += [
            (
                random.choice(choices["embed_dim"])
                if not sample_max
                else max(choices["embed_dim"])
            )
        ] * 3

    embed_dim += [embed_dim[-1]] * (depth - len(embed_dim))

    config["embed_dim"] = embed_dim
    config["layer_num"] = depth
    config["num_heads"] = choices["num_heads"][0]
    config["prune_granularity"] = np.ones((batch_size, depth))
    config["merge_granularity"] = np.ones((batch_size, depth))

    t = ["mlp_ratio", "embed_dim"]
    for d in t:
        config[d] = np.array(config[d])
        config[d] = np.tile(config[d], (batch_size, 1))

    return config


def max_config(choices, batch_size):
    config = {}

    config["layer_num"] = choices["depth"][0]
    config["num_heads"] = choices["num_heads"][0]
    config["mlp_ratio"] = np.ones((batch_size, 12)).astype(int) * max(
        choices["mlp_ratio"]
    )
    config["embed_dim"] = np.ones((batch_size, 12)).astype(int) * max(
        choices["embed_dim"]
    )
    config["prune_granularity"] = np.ones((batch_size, 12))
    config["merge_granularity"] = np.ones((batch_size, 12))

    return config


def min_config(choices, batch_size):
    config = {}

    config["layer_num"] = choices["depth"][0]
    config["num_heads"] = choices["num_heads"][0]
    config["mlp_ratio"] = np.ones((batch_size, 12)).astype(int) * min(
        choices["mlp_ratio"]
    )
    config["embed_dim"] = np.ones((batch_size, 12)).astype(int) * min(
        choices["embed_dim"]
    )
    config["prune_granularity"] = np.ones((batch_size, 12))
    config["merge_granularity"] = np.ones((batch_size, 12))

    return config


def gelu(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.nn.functional, "gelu"):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Vision_TransformerSuper(nn.Module):
    def __init__(
        self,
        cfg,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        pre_norm=True,
        scale=False,
        gp=False,
        relative_position=False,
        change_qkv=False,
        abs_pos=True,
        max_relative_position=14,
        mask_training=False,
        token_mode_str='pruning'
    ):
        super(Vision_TransformerSuper, self).__init__()

        self.mask_training = mask_training
        self.token_mode = _map_token_mode(token_mode_str)

        self.super_embed_dim = embed_dim
        self.super_mlp_ratio = mlp_ratio
        self.super_layer_num = depth
        self.super_num_heads = num_heads
        self.super_dropout = drop_rate
        self.super_attn_dropout = attn_drop_rate
        self.num_classes = num_classes
        self.pre_norm = pre_norm
        self.scale = scale
        self.patch_embed_super = PatchembedSuper(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.gp = gp

        # configs for the sampled subTransformer
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_layer_num = None
        self.sample_num_heads = None
        self.sample_dropout = None
        self.sample_output_dim = None

        self.blocks = nn.ModuleList()
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        
        N = 3
        decision_loc, linear_mapping_loc = [], []
        for i in range(depth):
            if i != 0 and i != depth-1 and i % N == 0:
                decision_loc.append(i)
            if (i + 1) % N == 0 and i != depth-1:
                linear_mapping_loc.append(i)
        
        self.decision_loc = decision_loc

        for i in range(depth):

            self.blocks.append(
                TransformerEncoderLayer(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    pre_norm=pre_norm,
                    scale=self.scale,
                    change_qkv=change_qkv,
                    relative_position=relative_position,
                    max_relative_position=max_relative_position,
                    id=i,
                    token_mode=self.token_mode,
                    decision_loc=decision_loc,
                    linear_mapping_loc=linear_mapping_loc
                )
            )

        # parameters for vision transformer
        num_patches = self.patch_embed_super.num_patches

        # Position encoding, trunc_normal is the initialization truncation function
        self.abs_pos = abs_pos
        if self.abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            trunc_normal_(self.pos_embed, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)
        if self.pre_norm:
            self.norm = LayerNormSuper(super_embed_dim=embed_dim)

        # classifier head
        self.head = (
            LinearSuper(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        self.apply(self._init_weights)

        self.iterator = None
        self.cfg = cfg

        # PPO initialization        
        self.selector = None
        self.use_selector = False
        
    @property
    def is_dynamic_model(self):
        return self.use_selector
    
    @property
    def token_optim_mode(self):
        return self.token_mode
    
    @property
    def is_batch_inference(self):
        return (self.mask_training and not self.training)
    
    def disable_selector(self):
        def _disabler(m):
            if hasattr(m, 'use_selector'):
                m.use_selector = False
        
        self.apply(_disabler)
    
    def enable_selector(self):
        def _enabler(m):
            if hasattr(m, 'use_selector'):
                m.use_selector = True
        
        self.apply(_enabler)
    
    def do_batch_inference(self):
        assert not self.training
        self.mask_training = False

    def do_masking_training(self):
        assert self.training
        self.mask_training = True
    
    def init_PPO_selector(
        self,
        batch_size,
        min_flag=False,
        max_flag=False,
        special_flag=False, 
        state_dim=197, 
        hidden_dim=256, 
        gamma=0.9, 
        lmbda=0.9, 
        epochs=10, 
        eps=0.2, 
        actor_lr=1e-4, 
        total_train_step=8000
    ):
        critic_lr = 50 * actor_lr

        if _check_is_token_pruning(self.token_optim_mode) or _check_is_token_merging(self.token_optim_mode):
            action_dim = 7
        elif _check_is_token_pruning_then_merging(self.token_optim_mode):
            action_dim = 8
        else:
            raise NotImplementedError

        self.selector = PPO(
            state_dim,
            hidden_dim,
            action_dim,
            actor_lr,
            critic_lr,
            lmbda,
            epochs,
            eps,
            gamma,
            total_train_step,
            token_mode=self.token_optim_mode,
        )
        self.selector.actor.eval()
        self.selector.critic.eval()

        choices = {
            "num_heads": self.cfg.SEARCH_SPACE.NUM_HEADS,
            "mlp_ratio": self.cfg.SEARCH_SPACE.MLP_RATIO,
            "embed_dim": self.cfg.SEARCH_SPACE.EMBED_DIM,
            "depth": self.cfg.SEARCH_SPACE.DEPTH,
        }
        self.choices = choices

        sample_config = sample_configs(
            choices=choices,
            batch_size=batch_size,
            sample_max=False,
        )
        self.sample_config = sample_config
        self._sample_one_subnets(batch_size)
        self.selector.actor.to(self.head.weight.device)
        self.selector.critic.to(self.head.weight.device)
        self.enable_selector()

        if min_flag:
            self.init_config["mlp_ratio"][:, 0:3] = np.ones(
                (batch_size, 3)
            ).astype(int) * min(choices["mlp_ratio"])
            self.init_config["embed_dim"][:, 0:3] = np.ones(
                (batch_size, 3)
            ).astype(int) * min(choices["embed_dim"])
            print("Minimum structure selected.")

        if max_flag:
            self.init_config["mlp_ratio"][:, 0:3] = np.ones(
                (batch_size, 3)
            ).astype(int) * max(choices["mlp_ratio"])
            self.init_config["embed_dim"][:, 0:3] = np.ones(
                (batch_size, 3)
            ).astype(int) * max(choices["embed_dim"])
            print("Maximum structure selected.")

        if special_flag and 216 in choices["embed_dim"]:
            self.init_config["mlp_ratio"][:, 0:3] = (
                np.ones((batch_size, 3)).astype(int) * 4
            )
            self.init_config["embed_dim"][:, 0:3] = (
                np.ones((batch_size, 3)).astype(int) * 216
            )
            print("Fixed structure selected.")

    def _sample_one_subnets(self, batch_size):
        self.init_config = max_config(self.choices, batch_size)

        self.init_config["mlp_ratio"][:, 0:3] = self.sample_config["mlp_ratio"][:, 0:3]
        self.init_config["embed_dim"][:, 0:3] = self.sample_config["embed_dim"][:, 0:3]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "rel_pos_embed"}

    def get_classifier(self):
        return self.head

    def set_sample_config(self, config: dict):
        if "prune_granularity" in config:
            self.sample_embed_dim = config["embed_dim"]
            self.sample_mlp_ratio = config["mlp_ratio"]
            self.sample_layer_num = config["layer_num"]
            self.sample_num_heads = config["num_heads"]
            self.prune_granularity = config["prune_granularity"]
            self.merge_granularity = config["merge_granularity"]

            self.sample_dropout = calc_dropout(
                self.super_dropout, self.sample_embed_dim[:, 0], self.super_embed_dim
            )

            self.sample_output_dim = np.hstack(
                (
                    self.sample_embed_dim[:, 1:],
                    self.sample_embed_dim[:, -1][:, np.newaxis],
                )
            )

    def set_sample_config_batch(self, config: dict, batch=64):
        # TODO: 未完成
        if "prune_granularity" in config:

            def avg_decision_across_batch(x, cfg=None):

                if self.training:
                    return x

                dtype = x.dtype

                x = np.mean(x, axis=0)

                if cfg:
                    x = x.round()
                    cfg = np.array(cfg)
                    for i in range(len(x)):

                        x[i] = cfg[np.argmin(np.abs(cfg - np.repeat(x[i], len(cfg))))]

                x = np.tile(x, [batch, 1]).astype(dtype)
                return x

            self.sample_embed_dim = avg_decision_across_batch(
                config["embed_dim"], self.cfg.SEARCH_SPACE.EMBED_DIM
            )
            # print(config["embed_dim"].shape)
            self.sample_mlp_ratio = avg_decision_across_batch(
                config["mlp_ratio"], self.cfg.SEARCH_SPACE.MLP_RATIO
            )
            self.sample_layer_num = config["layer_num"]
            self.sample_num_heads = config["num_heads"]
            self.prune_granularity = avg_decision_across_batch(
                config["prune_granularity"]
            )
            self.merge_granularity = config["merge_granularity"]

            self.sample_dropout = calc_dropout(
                self.super_dropout, self.sample_embed_dim[:, 0], self.super_embed_dim
            )

            self.sample_output_dim = np.hstack(
                (
                    self.sample_embed_dim[:, 1:],
                    self.sample_embed_dim[:, -1][:, np.newaxis],
                )
            )

    def get_sampled_params_numel(self, config):
        # return the params number of the sampled subnets
        numels = []
        for name, module in self.named_modules():
            if hasattr(module, "calc_sampled_param_num"):
                if (
                    name.split(".")[0] == "blocks"
                    and int(name.split(".")[1]) >= config["layer_num"]
                ):
                    continue
                numels.append(module.calc_sampled_param_num())

        embed_numel = self.sample_embed_dim[:, 0] * (
            2 + self.patch_embed_super.num_patches
        )

        other_numel = np.array(numels)
        other_numel = np.sum(other_numel, axis=0)

        result = np.round(embed_numel + other_numel).astype(int)

        return result

    def get_complexity(self, sequence_length=14*14):
        total_flops = self.patch_embed_super.get_complexity(sequence_length)

        zoom_ration = self.sample_embed_dim[:, 0] / self.super_embed_dim
        pos_total_flops = np.prod(self.pos_embed.size()) / 2.0
        total_flops += pos_total_flops * zoom_ration

        #############################################################################################
        # sequence_length changes caused by token pruning
        for blk in self.blocks:
            total_flops += blk.get_complexity()
        total_flops += self.head.get_complexity(blk.token_length_before)
        #############################################################################################

        return total_flops

    def forward_features(self, x, config: dict):
        B = x.shape[0]

        if self.is_batch_inference:
            self.set_sample_config_batch(config, B)
        else:
            self.set_sample_config(config)

        # TODO: 未完成
        x = self.patch_embed_super(
            x, self.sample_embed_dim[:, 0], batch_inference=self.is_batch_inference
        )

        cls_tokens = self.cls_token.expand(B, -1, -1)
        pos_embed = self.pos_embed

        if self.is_batch_inference:
            cls_tokens = cls_tokens[:, :, : self.sample_embed_dim[:, 0][0]]
            pos_embed = self.pos_embed[:, :, : self.sample_embed_dim[:, 0][0]]

        x = torch.cat((cls_tokens, x), dim=1)

        if self.abs_pos:
            x = x + pos_embed

        x = F.dropout(x, p=self.sample_dropout, training=self.training)

        ################################################################################
        # change the frequency of prune and merge
        original_token_idx = None
        token_length = None
        token_mask = None
        token_size = None

        for i, blk in enumerate(self.blocks):
            ################################################################################
            if i in self.decision_loc:
                re_info["cls_token"] = x[:, 0]

                state = torch.mean(re_info["k"], dim=-1)

                # state = torch.mean(re_info["q"], dim=-1)
                # state = torch.mean(re_info["v"], dim=-1)

                if self.is_batch_inference:
                    if state.shape[-1] < self.selector.state_dim:
                        diff = self.selector.state_dim - state.shape[-1]
                        state = F.pad(state, (0, diff), mode="constant", value=0)

                pre_action, _ = self.selector.take_action(state)

                config = self.selector.action_to_config(
                    pre_action,
                    self.choices,
                    config,
                    block_num=i,
                )

                if self.is_dynamic_model:
                    if self.is_batch_inference:
                        self.set_sample_config_batch(config, B)
                    else:
                        self.set_sample_config(config)
            ################################################################################

            sample_dropout = calc_dropout(
                self.super_dropout, self.sample_embed_dim[:, i], self.super_embed_dim
            )
            sample_attn_dropout = calc_dropout(
                self.super_attn_dropout,
                self.sample_embed_dim[:, i],
                self.super_embed_dim,
            )
            x, token_length, token_mask, token_size, re_info, original_token_idx = blk(
                x,
                is_identity_layer=False,
                prune_granularity=self.prune_granularity[:, i],
                merge_granularity=self.merge_granularity[:, i],
                sample_embed_dim=self.sample_embed_dim[:, i],
                sample_mlp_ratio=self.sample_mlp_ratio[:, i],
                sample_num_heads=self.sample_num_heads,
                sample_dropout=sample_dropout,
                sample_out_dim=self.sample_output_dim[:, i],
                sample_attn_dropout=sample_attn_dropout,
                original_token_idx=original_token_idx,
                token_length=token_length,
                token_mask=token_mask,
                token_size=token_size,
                batch_inference=self.is_batch_inference,
            )
        ################################################################################
        if self.pre_norm:
            x = self.norm(
                x,
                self.sample_embed_dim[:, -1],
                token_mask,
                batch_inference=self.is_batch_inference,
            )

        if self.gp:
            return torch.mean(x[:, 1:], dim=1)

        self.original_token_idx = original_token_idx
        self.batch_config = config  # for testing

        # TODO:待删除
        # print("最终剩余的token比例为:", x.shape[1] / 197)

        return x[:, 0]

    def forward(self, x):
        config = copy.deepcopy(self.init_config)

        x = self.forward_features(x, config)
        x = self.head(
            x,
            self.sample_embed_dim[:, -1],
            np.full(self.sample_embed_dim[:, -1].shape, self.num_classes),
            batch_inference=self.is_batch_inference,
        )

        return x


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments which
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        dropout=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        pre_norm=True,
        scale=False,
        relative_position=False,
        change_qkv=False,
        max_relative_position=14,
        id=0,
        token_mode=None,
        decision_loc=None,
        linear_mapping_loc=None
    ):
        super().__init__()

        # the configs of super arch of the encoder, three dimension [embed_dim, mlp_ratio, and num_heads]
        self.super_embed_dim = dim
        self.super_mlp_ratio = mlp_ratio
        self.super_ffn_embed_dim_this_layer = int(mlp_ratio * dim)
        self.super_num_heads = num_heads
        self.normalize_before = pre_norm
        self.super_dropout = attn_drop
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale = scale
        self.relative_position = relative_position
        self.id = id
        self.token_mode = token_mode
        # self.super_activation_dropout = getattr(args, 'activation_dropout', 0)

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_num_heads_this_layer = None
        self.sample_scale = None
        self.sample_dropout = None
        self.sample_attn_dropout = None

        self.token_length_before = None
        self.token_length_after = None
        self.original_token_idx = None

        self.is_identity_layer = None
        self.attn = AttentionSuper(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=dropout,
            scale=self.scale,
            relative_position=self.relative_position,
            change_qkv=change_qkv,
            max_relative_position=max_relative_position,
        )

        self.attn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.ffn_layer_norm = LayerNormSuper(self.super_embed_dim)
        # self.dropout = dropout
        self.activation_fn = gelu
        # self.normalize_before = args.encoder_normalize_before

        self.fc1 = LinearSuper(
            super_in_dim=self.super_embed_dim,
            super_out_dim=self.super_ffn_embed_dim_this_layer,
        )
        self.fc2 = LinearSuper(
            super_in_dim=self.super_ffn_embed_dim_this_layer,
            super_out_dim=self.super_embed_dim,
        )

        if id in linear_mapping_loc:
            self.lln = LinearSuper(
                super_in_dim=self.super_embed_dim, super_out_dim=self.super_embed_dim
            )
        else:
            self.lln = nn.Identity()
        
        self.decision_loc = decision_loc
        self.use_selector = False

    @property
    def is_dynamic_model(self):
        return self.use_selector
    
    def set_sample_config(
        self,
        is_identity_layer,
        sample_embed_dim=None,
        sample_mlp_ratio=None,
        sample_num_heads=None,
        sample_dropout=None,
        sample_attn_dropout=None,
        sample_out_dim=None,
    ):
        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False

        self.sample_embed_dim = sample_embed_dim
        self.sample_out_dim = sample_out_dim
        self.sample_mlp_ratio = sample_mlp_ratio
        self.sample_ffn_embed_dim_this_layer = (
            sample_embed_dim * sample_mlp_ratio
        ).astype(int)
        self.sample_num_heads_this_layer = sample_num_heads

        self.sample_dropout = sample_dropout
        self.sample_attn_dropout = sample_attn_dropout

    def gather_vectors_by_index(self, data, idx):
        # Used to maintain the order of the original tokens
        if idx.shape[1] < data.shape[1]:
            sub_data = copy.deepcopy(data[:, : idx.shape[1]])
            replace_data = torch.gather(sub_data, dim=1, index=idx)
            data[:, : idx.shape[1]] = replace_data

            result = data
        else:
            result = torch.gather(data, dim=1, index=idx)

        return result

    def get_batch_merge_func(
        self,
        metric: torch.Tensor,
        merge_granularity,
        token_length_before,
    ):
        with torch.no_grad():
            # Calculate the number of tokens to keep in each batch
            kept_number = (merge_granularity * token_length_before).astype(int)[0]

            metric = metric / metric.norm(dim=-1, keepdim=True)
            unimportant_tokens_metric = metric[:, kept_number:]

            compress_number = unimportant_tokens_metric.shape[1]

            important_tokens_metric = metric[:, :kept_number]

            similarity = unimportant_tokens_metric @ important_tokens_metric.transpose(
                -1, -2
            )

            similarity[..., :, 0] = -math.inf

            node_max, node_idx = similarity.max(dim=-1)
            dst_idx = node_idx[..., None]

        def batch_merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            src = x[:, kept_number:]
            dst = x[:, :kept_number]
            n, t1, c = src.shape
            dst = dst.scatter_reduce(
                -2, dst_idx.expand(n, compress_number, c), src, reduce=mode
            )

            return dst

        return batch_merge, kept_number

    def get_merge_func(
        self,
        metric: torch.Tensor,
        merge_granularity,
        token_length_before,
    ):
        B, N, C = metric.shape

        with torch.no_grad():
            metric = metric / metric.norm(dim=-1, keepdim=True)

            # Calculate the number of tokens to keep in each batch
            token_length_after = (merge_granularity * token_length_before).astype(int)

            # Generate the matrix of unimportant elements
            unimportant_mask_indices = torch.arange(N).unsqueeze(0).unsqueeze(
                -1
            ) >= torch.tensor(token_length_after).unsqueeze(1).unsqueeze(2)
            unimportant_token_mask = (
                unimportant_mask_indices.expand(B, N, C).float().to("cuda")
            )

            # Copy a mask that is not used to calculate correlation
            raw_unimportant_token_mask = unimportant_token_mask.clone()
            unimportant_token_mask[unimportant_token_mask == 0] = math.inf

            unimportant_tokens_metric = metric * unimportant_token_mask
            # Correct the metric interference caused by multiplication
            unimportant_tokens_metric[unimportant_tokens_metric == -math.inf] = math.inf
            unimportant_tokens_metric[torch.isnan(unimportant_tokens_metric)] = math.inf

            # Generate the matrix of important elements
            important_mask_indices = torch.arange(N).unsqueeze(0).unsqueeze(
                -1
            ) < torch.tensor(token_length_after).unsqueeze(1).unsqueeze(2)
            important_token_mask = (
                important_mask_indices.expand(B, N, C).float().to("cuda")
            )
            raw_important_token_mask = important_token_mask.clone()
            important_token_mask[important_token_mask == 0] = math.inf
            important_tokens_metric = metric * important_token_mask
            important_tokens_metric[important_tokens_metric == -math.inf] = math.inf
            important_tokens_metric[torch.isnan(important_tokens_metric)] = math.inf

            # Calculate the similarity between important and unimportant tokens
            similarity = unimportant_tokens_metric @ important_tokens_metric.transpose(
                -1, -2
            )

            # Replace inf in the matrix with -inf, note that there will be nan, which needs to be handled
            similarity[torch.isnan(similarity)] = -math.inf
            similarity[similarity == math.inf] = -math.inf

            # Mark the cls token as irrelevant to ensure it is not merged
            similarity[..., :, 0] = -math.inf

            node_max, node_idx = similarity.max(dim=-1)

            # Change the correlation of the masked tokens to merge them with the least important token,
            # which will be masked later. Cannot use -1 as scatter function cannot handle it.
            # This vector stores the index of the most relevant important token for each unimportant token.
            node_idx[node_idx == 0] = N - 1

            dst_idx = node_idx[..., None]

            def size_merge(
                size: torch.tensor, import_mask, unimport_mask, mode="sum"
            ) -> torch.Tensor:
                import_mask = import_mask[:, :, -1:]
                unimport_mask = unimport_mask[:, :, -1:]

                src = size * unimport_mask  # unimportant token
                dst = size * import_mask  # important token

                n, t1, c = src.shape

                dst = dst.scatter_reduce(-2, dst_idx.expand(n, t1, c), src, reduce=mode)

                # re mark
                dst = dst * import_mask
                dst[dst == 0] = 1

                return dst

            def merge(
                x: torch.tensor, import_mask, unimport_mask, mode="mean"
            ) -> torch.Tensor:
                # Align the dimensions of the mask
                if import_mask.shape[-1] != x.shape[-1]:
                    extension = import_mask[:, :, -1:].repeat(
                        1, 1, x.shape[-1] - import_mask.shape[-1]
                    )
                    import_mask = torch.cat((import_mask, extension), dim=2)

                    extension_2 = unimport_mask[:, :, -1:].repeat(
                        1, 1, x.shape[-1] - unimport_mask.shape[-1]
                    )
                    unimport_mask = torch.cat((unimport_mask, extension_2), dim=2)

                src = x * unimport_mask  # unimportant token
                dst = x * import_mask  # important token

                n, t1, c = src.shape

                dst = dst.scatter_reduce(-2, dst_idx.expand(n, t1, c), src, reduce=mode)
                dst = dst * import_mask

                return dst, import_mask

            return (
                merge,
                size_merge,
                token_length_after,
                raw_important_token_mask,
                raw_unimportant_token_mask,
            )

    def forward(
        self,
        x,
        is_identity_layer,
        prune_granularity=1,
        merge_granularity=1,
        sample_embed_dim=None,
        sample_mlp_ratio=None,
        sample_num_heads=None,
        sample_dropout=None,
        sample_attn_dropout=None,
        sample_out_dim=None,
        original_token_idx=None,
        token_length=None,
        token_mask=None,
        token_size=None,
        batch_inference=False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(batch, patch_num , sample_embed_dim)`

        Returns:
            encoded output of shape `(batch, patch_num, sample_embed_dim)`
        """

        B, N, C = x.shape

        if self.is_identity_layer:
            return x

        self.set_sample_config(
            is_identity_layer,
            sample_embed_dim,
            sample_mlp_ratio,
            sample_num_heads,
            sample_dropout,
            sample_attn_dropout,
            sample_out_dim,
        )

        if self.id == 0:
            self.token_mask = None
            self.token_size = torch.ones([B, N, 1], device=x.device)
        else:
            self.token_mask = token_mask
            self.token_size = token_size

        # print("检查")
        # print("self.id: ", self.id)

        residual = x
        x = self.maybe_layer_norm(
            self.attn_layer_norm,
            x,
            self.token_mask,
            before=True,
            batch_inference=batch_inference,
        )

        # return the attention matrix for token pruning
        x, re_info = self.attn(
            x,
            sample_q_embed_dim=self.sample_num_heads_this_layer * 64,
            sample_num_heads=self.sample_num_heads_this_layer,
            sample_in_embed_dim=self.sample_embed_dim,
            token_length=token_length,
            token_size=self.token_size,
            batch_inference=batch_inference,
        )
        attn = re_info["attn"]

        x = F.dropout(x, p=self.sample_attn_dropout, training=self.training)
        x = self.drop_path(x)

        if batch_inference:
            residual = residual[:, :, : sample_embed_dim[0]]
        x = residual + x

        if self.id == 0:
            self.token_length_before = np.ones(B) * N
            self.token_length_after = self.token_length_before

            # Generate token order list
            array = [[i for i in range(x.shape[1])] for _ in range(x.shape[0])]
            self.original_token_idx = torch.tensor(array).to("cuda")

        elif self.id not in self.decision_loc:
            self.token_length_before = token_length
            self.token_length_after = self.token_length_before

            self.original_token_idx = original_token_idx

        else:
            self.token_length_before = token_length
            self.original_token_idx = original_token_idx

            ###############################################################################
            # Calculate the importance of tokens to cls, measured using the attention matrix
            # The dimension of cls_attn：[B, sample_num_heads, N, N]
            
            if self.is_dynamic_model:
            
                cls_attn = attn[:, :, 0, 1:]
                cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
                _, idx = torch.sort(
                    cls_attn, descending=True
                )  # The dimension of idx: [B, N-1]

                self.original_token_idx = self.gather_vectors_by_index(
                    self.original_token_idx, idx
                )

                cls_index = torch.zeros((B, 1), device=idx.device).long()
                # Set the importance of cls_token to the highest
                idx = torch.cat((cls_index, idx + 1), dim=1)

                # Sort tokens by importance
                x = torch.gather(
                    x, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
                )
                # token_size also needs to be sorted by importance
                self.token_size = torch.gather(
                    self.token_size, dim=1, index=idx.unsqueeze(-1)
                )

                ###############################################################################
                if _check_is_token_pruning(self.token_mode) or _check_is_token_pruning_then_merging(self.token_mode):
                    # token pruning
                    num_tokens_to_keep = (
                        prune_granularity * self.token_length_before
                    ).astype(int)

                    if batch_inference:
                        x = x[:, : num_tokens_to_keep[0]]
                        self.token_size = self.token_size[:, : num_tokens_to_keep[0]]
                    else:
                        # add mask
                        mask_indices = torch.arange(N).unsqueeze(0).unsqueeze(
                            -1
                        ) < torch.tensor(num_tokens_to_keep).unsqueeze(1).unsqueeze(2)
                        self.token_mask = mask_indices.expand(B, N, C).float().to("cuda")

                        x *= self.token_mask

                    # save the token number after pruning
                    self.token_length_after = num_tokens_to_keep

                # connect block
                if _check_is_token_pruning_then_merging(self.token_mode):
                    self.token_length_before = self.token_length_after

                ###############################################################################
                if _check_is_token_merging(self.token_mode) or _check_is_token_pruning_then_merging(self.token_mode):
                    if batch_inference:
                        batch_merge, self.token_length_after = self.get_batch_merge_func(
                            x,
                            merge_granularity,
                            token_length_before=self.token_length_before,
                        )
                        x = batch_merge(x)
                        self.token_size = batch_merge(self.token_size, mode="sum")
                    else:
                        # Merging
                        (
                            merge,
                            size_merge,
                            self.token_length_after,
                            import_mask,
                            unimport_mask,
                        ) = self.get_merge_func(
                            x,
                            # re_info["k"],
                            merge_granularity=merge_granularity,
                            token_length_before=self.token_length_before,
                        )
                        x, self.token_mask = merge(x, import_mask, unimport_mask)
                        self.token_size = size_merge(
                            self.token_size, import_mask, unimport_mask
                        )
            ###############################################################################
        x = self.maybe_layer_norm(
            self.attn_layer_norm,
            x,
            self.token_mask,
            after=True,
            batch_inference=batch_inference,
        )
        residual = x

        # print("residual.shape: ", residual.shape)

        x = self.maybe_layer_norm(
            self.ffn_layer_norm,
            x,
            self.token_mask,
            before=True,
            batch_inference=batch_inference,
        )

        x = self.activation_fn(
            self.fc1(
                x,
                sample_in_dim=self.sample_embed_dim,
                sample_out_dim=self.sample_ffn_embed_dim_this_layer,
                token_length=self.token_length_after,
                batch_inference=batch_inference,
            )
        )
        x = F.dropout(x, p=self.sample_dropout, training=self.training)

        x = self.fc2(
            x,
            sample_in_dim=self.sample_ffn_embed_dim_this_layer,
            sample_out_dim=self.sample_out_dim,
            token_length=self.token_length_after,
            batch_inference=batch_inference,
        )
        x = F.dropout(x, p=self.sample_dropout, training=self.training)

        if self.scale:
            x = x * (self.super_mlp_ratio / self.sample_mlp_ratio)
        x = self.drop_path(x)

        # print("x.shape: ", x.shape)
        # print("self.sample_embed_dim: ", self.sample_embed_dim[0])
        # print("self.sample_out_dim: ", self.sample_out_dim[0])

        if isinstance(self.lln, LinearSuper):
            x = (
                self.lln(
                    residual,
                    # TODO:需要全部更改一下
                    # sample_in_dim=self.sample_ffn_embed_dim_this_layer,
                    sample_in_dim=self.sample_embed_dim,
                    sample_out_dim=self.sample_out_dim,
                    token_length=self.token_length_after,
                    batch_inference=batch_inference,
                )
                + x
            )
        else:
            # x = self.lln(residual) + x
            if batch_inference:
                residual = self.lln(residual)
                # Check if the last dimension of residual and x are consistent
                if residual.shape[-1] != x.shape[-1]:
                    if residual.shape[-1] < x.shape[-1]:
                        # If the last dimension of residual is smaller than x, pad with zeros
                        diff = x.shape[-1] - residual.shape[-1]
                        residual = F.pad(
                            residual, (0, diff)
                        )  # Pad only the last dimension with zeros
                    else:
                        # If the last dimension of residual is larger than x, trim to match x
                        residual = residual[
                            ..., : x.shape[-1]
                        ]  # Trim to match the last dimension of x
                x = residual + x
            else:
                x = self.lln(residual) + x

        x = self.maybe_layer_norm(
            self.ffn_layer_norm,
            x,
            self.token_mask,
            after=True,
            batch_inference=batch_inference,
        )

        re_info["token_length"] = self.token_length_after

        return (
            x,
            self.token_length_after,
            self.token_mask,
            self.token_size,
            re_info,
            self.original_token_idx,
        )

    def maybe_layer_norm(
        self,
        layer_norm,
        x,
        token_mask,
        before=False,
        after=False,
        batch_inference=False,
    ):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x, self.sample_embed_dim, token_mask, batch_inference)
        else:
            return x

    def get_complexity(self):
        if self.is_identity_layer:
            return 0

        #############################################################################################
        # Calculate the model FLOPs based on the number of tokens before and after pruning
        # Steps before pruning
        total_flops = self.attn_layer_norm.get_complexity(self.token_length_before)
        total_flops += self.attn.get_complexity(self.token_length_before)

        # Steps after pruning
        total_flops += self.ffn_layer_norm.get_complexity(self.token_length_after)
        total_flops += self.fc1.get_complexity(self.token_length_after)
        total_flops += self.fc2.get_complexity(self.token_length_after)
        #############################################################################################

        return total_flops


def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    return dropout * 1.0 * sample_embed_dim[0] / super_embed_dim
