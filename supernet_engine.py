import math
import sys
from typing import Iterable, Optional
from timm.utils.model import unwrap_model
import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from lib import utils
import random
import time
import numpy as np
from lib.dataloader import *
from lib.mixup import NLLMultiLabelSmooth, MixUpWrapper


def sample_configs(choices, op=random.choice):
    # This function is used to select specific network parameters during training
    config = {}
    dimensions = ["mlp_ratio", "num_heads"]

    # choose depth
    depth = op(choices["depth"])

    # Select mlp_ratio and num_heads, note that these two dimensions are independently chosen for each attention module
    for dimension in dimensions:
        config[dimension] = [(op(choices[dimension])) for _ in range(depth)]

    # Select embed_dim, the same choice is made for every 3 attention modules, and the parameters of the extra single attention module are the same as the last group. Note that the search space is expanded here.
    embed_dim = []
    group = depth // 3
    for _ in range(group):
        embed_dim += [(op(choices["embed_dim"]))] * 3

    embed_dim += [embed_dim[-1]] * (depth - len(embed_dim))
    config["embed_dim"] = embed_dim

    # config['embed_dim'] = [random.choice(choices['embed_dim']) if not sample_max else max(choices['embed_dim'])]*depth

    # config = {'mlp_ratio': [2, 4, 2, 6, 4, 2, 4, 4, 6, 4, 6, 6], 'num_heads': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'embed_dim': [240, 240, 240, 176, 176, 176, 176, 176, 176, 176, 176, 176], 'layer_num': 12}
    config["layer_num"] = depth

    return config


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    amp: bool = True,
    teacher_model: torch.nn.Module = None,
    teach_loss: torch.nn.Module = None,
    choices=None,
    mode="super",
    retrain_config=None,
    args=None,
    data_loader=None,
):

    model.train()
    criterion.train()

    # set random seed
    random.seed(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    if mode == "retrain":
        config = retrain_config
        model_module = unwrap_model(model)
        print(config)
        model_module.set_sample_config(config=config)
        print(model_module.get_sampled_params_numel(config))

    train_iter = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # sample random config
        optimizer.zero_grad()

        for _ in range(1):

            config = sample_configs(choices=choices)
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
            model_module._sample_one_subnets()

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)

            if amp:
                with torch.cuda.amp.autocast():
                    if teacher_model:
                        with torch.no_grad():
                            teach_output = teacher_model(samples)
                        _, teacher_label = teach_output.topk(1, 1, True, True)
                        outputs = model(samples)
                        loss = 1 / 2 * criterion(outputs, targets) + 1 / 2 * teach_loss(
                            outputs, teacher_label.squeeze()
                        )
                    else:
                        outputs = model(samples)
                        loss = criterion(outputs, targets)
            else:
                outputs = model(samples)
                if teacher_model:
                    with torch.no_grad():
                        teach_output = teacher_model(samples)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    loss = 1 / 2 * criterion(outputs, targets) + 1 / 2 * teach_loss(
                        outputs, teacher_label.squeeze()
                    )
                else:
                    loss = criterion(outputs, targets)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # this attribute is added by timm on one optimizer (adahessian)
            if amp:
                # is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                # loss_scaler(loss, optimizer, clip_grad=max_norm,
                #         parameters=model.parameters(), create_graph=is_second_order)
                loss_scaler._scaler.scale(loss).backward()

            else:
                loss.backward()
                optimizer.step()

        if amp:
            if max_norm is not None:
                loss_scaler._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            loss_scaler._scaler.step(optimizer)
            loss_scaler._scaler.update()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        train_iter += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    data_loader,
    model,
    device,
    amp=True,
    choices=None,
    mode="super",
    retrain_config=None,
):
    criterion = torch.nn.CrossEntropyLoss()

    # switch to evaluation mode
    total_acc = []
    total_flops = []
    total_param = []
    for i in range(5):
        model.eval()
        model_module = unwrap_model(model)

        batch_acc_list = []
        batch_flops_list = []
        batch_param_list = []

        batch_iter = 0
        for images, target in data_loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # compute output
            if amp:
                with torch.cuda.amp.autocast():
                    output = model(images)
                    loss = criterion(output, target)
            else:
                output = model(images)
                loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            flops = model_module.get_complexity(14 * 14)
            param = model_module.get_sampled_params_numel(
                config=model_module.batch_config
            )

            batch_acc = acc1.cpu().numpy()
            batch_flops = np.mean(flops / 1e9)
            batch_param = np.mean(param / 1e6)

            batch_acc_list.append(batch_acc)
            batch_flops_list.append(batch_flops)
            batch_param_list.append(batch_param)

            batch_iter += 1

            if batch_iter % 10 == 0:
                print(
                    "id:{}, batch_acc:{:.2f}, batch_Gflops:{:.2f}, batch_param(M):{:.2f}".format(
                        batch_iter,
                        batch_acc,
                        batch_flops,
                        batch_param,
                    )
                )

            if batch_iter == data_loader.loader_len - 1:
                data_loader.reset()
                break

        single_acc = np.mean(batch_acc_list)
        single_flops = np.mean(batch_flops_list)
        single_param = np.mean(batch_param_list)

        total_acc.append(single_acc)
        total_flops.append(single_flops)
        total_param.append(single_param)

        print("*" * 40)
        print("Single test results are shown as follows:")
        print(
            "single_acc:{:.2f}, single_Gflops:{:.2f}, single_param(M):{:.2f}".format(
                batch_acc,
                batch_flops,
                batch_param,
            )
        )
        print("*" * 40)

    print()
    print("*" * 40)
    print("The results of this test is:")
    print(
        "_acc:{:.2f}, avg_Gflops:{:.2f}, avg_param(M):{:.2f}".format(
            np.mean(total_acc),
            np.mean(total_flops),
            np.mean(total_param),
        )
    )
    print("*" * 40)
