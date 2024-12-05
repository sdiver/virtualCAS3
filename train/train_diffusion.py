"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
Face diffusion model
"""

import json
import os
import torch
import torch.multiprocessing as mp

from dataLoad.get_data import get_dataset_loader, load_local_data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from train.train_platforms import ClearmlPlatform, NoPlatform, TensorboardPlatform
from train.training_loop import TrainLoop
from utils.diff_parser_utils import train_args
from utils.misc import cleanup, fix_seed, setup_dist
from utils.model_util import create_model_and_diffusion


def main(rank: int, world_size: int):
    """
    这段代码定义了一个名为main的函数，用于分布式训练模型。以下是代码的详细解释：
    main
        Initializes and runs the training process.

    Parameters:
        rank (int): The rank of the current process in the distributed training setup.
        world_size (int): The total number of processes in the distributed training setup.
    """
    """
    1\参数设置与种子固定
    这两行代码调用了一些函数来解析训练参数并固定随机种子
    """
    args = train_args()
    fix_seed(args.seed)
    """
    2\训练平台与分布式设置
    使用解析出来的参数初始化训练平台，并进行分布式环境的设置。 
    """
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name="Args")
    setup_dist(rank, world_size,args.device)
    """
    3\主进程特权处理
    只有在rank为0时处理保存路径，确保保存路径存在且没有被覆盖。 
    """
    if rank == 0:
        if args.save_dir is None:
            raise FileNotFoundError("save_dir was not specified.")
        elif os.path.exists(args.save_dir) and not args.overwrite:
            raise FileExistsError("save_dir [{}] already exists., saving to new directory:".format(args.save_dir))
        elif not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args_path = os.path.join(args.save_dir, "args.json")
        with open(args_path, "w") as fw:
            json.dump(vars(args), fw, indent=4, sort_keys=True)
    """
    4\数据集加载
    加载数据集，检查数据路径是否存在。 
    """
    if not os.path.exists(args.data_root):
        args.data_root = args.data_root.replace("/home/", "/derived/")

    data_dict = load_local_data(args.data_root, audio_per_frame=1600)
    """
    5\日志记录与模型创建
    创建日志记录器，模型和扩散模型，并将模型分配到对应GPU。 
    """
    print("creating data loader...")
    data = get_dataset_loader(args=args, data_dict=data_dict)

    print("creating logger...")
    writer = SummaryWriter(args.save_dir)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, split_type="train")
    model.to(rank)
    if world_size > 1:
        model = DDP(
            model, device_ids=[rank], output_device=rank
        )
    """
    6\参数统计与训练
    打印模型参数数量，开始训练循环，最后关闭训练平台并清理资源。 
    """
    params = (
        model.module.parameters_w_grad()
        if world_size > 1
        else model.parameters_w_grad()
    )
    print("Total params: %.2fM" % (sum(p.numel() for p in params) / 1000000.0))
    print("Training...")

    TrainLoop(
        args, train_platform, model, diffusion, data, writer, rank, world_size
    ).run_loop()
    train_platform.close()
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"using {world_size} gpus")
    if world_size > 1:
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    else:
        main(rank=0, world_size=1)
