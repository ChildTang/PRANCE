# -*- coding: utf-8 -*-

import argparse
import numpy as np
import torch
import yaml
from pathlib import Path
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, ModelEma, accuracy
from lib.datasets import build_dataset
from lib import utils
from lib.config import supernet_cfg, update_config_from_file
from model.supernet_transformer import Vision_TransformerSuper
# from lib.dataloader import *


def get_args_parser():
    parser = argparse.ArgumentParser(
        "AutoFormer training and evaluation script", add_help=False
    )
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    # PRANCE: For batch inference
    parser.add_argument("--batch-inference", action="store_true")
    parser.set_defaults(batch_inference=False)
    # config file
    parser.add_argument(
        "--cfg",
        help="experiment configure file name",
        # required=True,
        default="structure_of_meta-network/supernet-T.yaml",
        type=str,
    )

    # custom parameters
    parser.add_argument(
        "--platform",
        default="pai",
        type=str,
        choices=["itp", "pai", "aml"],
        help="Name of model to train",
    )
    parser.add_argument(
        "--teacher_model", default="", type=str, help="Name of teacher model to train"
    )
    parser.add_argument("--relative_position", action="store_true")
    parser.add_argument("--gp", action="store_true")
    # parser.add_argument("--change_qkv", action="store_true")
    parser.add_argument("--change_qkv", default=True)
    parser.add_argument(
        "--max_relative_position",
        type=int,
        default=14,
        help="max distance in relative position embedding",
    )

    # Model parameters
    parser.add_argument(
        "--model", default="", type=str, metavar="MODEL", help="Name of model to train"
    )
    # AutoFormer config
    parser.add_argument(
        "--mode",
        type=str,
        default="super",
        choices=["super", "retrain"],
        help="mode of AutoFormer",
    )
    parser.add_argument("--input-size", default=224, type=int)
    parser.add_argument("--patch_size", default=16, type=int)

    parser.add_argument(
        "--drop",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--drop-path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )
    parser.add_argument(
        "--drop-block",
        type=float,
        default=None,
        metavar="PCT",
        help="Drop block rate (default: None)",
    )

    parser.add_argument(
        "--model-ema",
        action="store_true",
        default=False,
    )
    parser.add_argument("--no-model-ema", action="store_false", dest="model_ema")
    # parser.set_defaults(model_ema=True)
    parser.add_argument("--model-ema-decay", type=float, default=0.99996, help="")
    parser.add_argument(
        "--model-ema-force-cpu", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--rpe_type", type=str, default="bias", choices=["bias", "direct"]
    )
    parser.add_argument("--post_norm", action="store_true")
    parser.add_argument("--no_abs_pos", action="store_true")

    # Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt-eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt-betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    # Learning rate schedule parameters
    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "cosine"',
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--lr-noise",
        type=float,
        nargs="+",
        default=None,
        metavar="pct, pct",
        help="learning rate noise on/off epoch percentages",
    )
    parser.add_argument(
        "--lr-noise-pct",
        type=float,
        default=0.67,
        metavar="PERCENT",
        help="learning rate noise limit percent (default: 0.67)",
    )
    parser.add_argument(
        "--lr-noise-std",
        type=float,
        default=1.0,
        metavar="STDDEV",
        help="learning rate noise std-dev (default: 1.0)",
    )
    parser.add_argument(
        "--ppo_name",
        default="none",
        type=str,
    )
    parser.add_argument(
        "--ppo_path",
        default="none",
        type=str,
    )
    parser.add_argument(
        "--token_mode",
        default="none",
        type=str,
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )
    parser.add_argument(
        "--lr-power",
        type=float,
        default=1.0,
        help="power of the polynomial lr scheduler",
    )

    parser.add_argument(
        "--decay-epochs",
        type=float,
        default=30,
        metavar="N",
        help="epoch interval to decay LR",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=20,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--cooldown-epochs",
        type=int,
        default=10,
        metavar="N",
        help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
    )
    parser.add_argument(
        "--patience-epochs",
        type=int,
        default=10,
        metavar="N",
        help="patience epochs for Plateau LR scheduler (default: 10",
    )
    parser.add_argument(
        "--decay-rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )

    # Augmentation parameters
    parser.add_argument(
        "--color-jitter",
        type=float,
        default=0.4,
        metavar="PCT",
        help="Color jitter factor (default: 0.4)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)',
    ),
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )
    parser.add_argument(
        "--train-interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    parser.add_argument("--repeated-aug", action="store_true")
    parser.add_argument("--no-repeated-aug", action="store_false", dest="repeated_aug")

    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Mixup params
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.8,
        help="mixup alpha, mixup enabled if > 0. (default: 0.8)",
    )
    parser.add_argument(
        "--cutmix",
        type=float,
        default=1.0,
        help="cutmix alpha, cutmix enabled if > 0. (default: 1.0)",
    )
    parser.add_argument(
        "--cutmix-minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup-prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup-switch-prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup-mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # Dataset parameters
    parser.add_argument(
        "--data-path", default="./data/imagenet/", type=str, help="dataset path"
    )
    parser.add_argument(
        "--data-set",
        default="IMNET",
        choices=["CIFAR", "IMNET", "INAT", "INAT19"],
        type=str,
        help="Image Net dataset path",
    )
    parser.add_argument(
        "--inat-category",
        default="name",
        choices=[
            "kingdom",
            "phylum",
            "class",
            "order",
            "supercategory",
            "family",
            "genus",
            "name",
        ],
        type=str,
        help="semantic granularity",
    )

    parser.add_argument(
        "--output_dir",
        default="./test_li/",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--resume",
        default="out_dim_var/checkpoint.pth",
        help="resume from checkpoint",
    )
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument(
        "--eval",
        # action="store_true",
        default=True,
        help="Perform evaluation only",
    )
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument(
        "--dist-eval",
        # action="store_true",
        # default=False,
        default=True,
        help="Enabling distributed evaluation",
    )
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.set_defaults(amp=True)

    return parser


def main(args):
    update_config_from_file(args.cfg)

    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    device = torch.device(args.device)

    print("Creating SuperVisionTransformer")
    print(supernet_cfg)
    model = Vision_TransformerSuper(
        cfg=supernet_cfg,
        img_size=args.input_size,
        patch_size=args.patch_size,
        embed_dim=supernet_cfg.SUPERNET.EMBED_DIM,
        depth=supernet_cfg.SUPERNET.DEPTH,
        num_heads=supernet_cfg.SUPERNET.NUM_HEADS,
        mlp_ratio=supernet_cfg.SUPERNET.MLP_RATIO,
        qkv_bias=True,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        gp=args.gp,
        num_classes=1000,
        max_relative_position=args.max_relative_position,
        relative_position=args.relative_position,
        change_qkv=args.change_qkv,
        abs_pos=not args.no_abs_pos,
        mask_training=not args.batch_inference,
        token_mode_str=args.token_mode,
    )

    seq_length = args.input_size // args.patch_size * args.input_size // args.patch_size
    model.init_PPO_selector(
        args.batch_size,
        state_dim=seq_length + 1,  # +1 for cls token
        max_flag=True,
    )

    model.to(device)
    model.selector.actor.to(device)
    model.selector.critic.to(device)

    # path_name = args.ppo_path
    test_path = f"{args.ppo_path}/{args.ppo_name}"
    model.selector.load_model(path_name=test_path, file_name="")
    print("The PPO model is: ", test_path)

    model_ema = ModelEma(model=model, decay=0.99985, device="", resume="")

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    # save config for later experiments
    with open(output_dir / "config.yaml", "w") as f:
        f.write(args_text)

    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")

        model.load_state_dict(checkpoint["model"])

        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
        if args.model_ema:
            model_ema._load_weights(checkpoint["model_ema"])

    # model eval
    if args.eval:
        dataset_val, _ = build_dataset(is_train=False, args=args)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=int(args.batch_size),
            sampler=sampler_val,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        model.eval()
        print("*" * 60)
        print("Start evaluation!!!")
        print("*" * 60)

        print(model.is_batch_inference, model.is_dynamic_model)

        batch_acc_list = []
        batch_flops_list = []
        batch_param_list = []

        batch_iter = 0
        for images, target in data_loader_val:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                output = model(images)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            flops = model.get_complexity(14 * 14)
            param = model.get_sampled_params_numel(config=model.batch_config)

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

        single_acc = np.mean(batch_acc_list)
        single_flops = np.mean(batch_flops_list)
        single_param = np.mean(batch_param_list)

        print("*" * 40)
        print("Single test results are as follows:")
        print(
            "single_acc:{:.2f}, single_Gflops:{:.2f}, single_param(M):{:.2f}".format(
                single_acc,
                single_flops,
                single_param,
            )
        )
        print("*" * 40)

    print("The PPO model is: ", args.ppo_name)
    print("resume: ", args.resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Prance training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir + "/" + args.ppo_name

    args.resume = args.output_dir + "/checkpoint.pth"
    print("Current token compression strategy is: ", args.token_mode)

    main(args)