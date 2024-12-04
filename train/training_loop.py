"""
版权所有 (c) Meta Platforms, Inc. 及其附属公司。
保留所有权利。
本源代码根据本源代码树根目录中的 LICENSE 文件中的许可证授权。
"""

# 导入必要的库
import cProfile as profile
import functools
import pstats
from typing import Dict, Any

import blobfile as bf
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

import utils.logger as logger
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, create_named_schedule_sampler
from utils.misc import dev, load_state_dict

# 初始日志损失尺度
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        args: Any,
        train_platform: Any,
        model: torch.nn.Module,
        diffusion: Any,
        data: Any,
        writer: Any,
        rank: int = 0,
        world_size: int = 1
    ):
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        # 初始化训练循环的各种参数
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        self.cond_mode = model.module.cond_mode if world_size > 1 else model.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps
        self.rank = rank
        self.world_size = world_size

        # 初始化步骤计数器和相关变量
        self.step = 0  # 当前训练步骤
        self.resume_step = 0  # 恢复训练时的起始步骤
        self.global_batch = self.batch_size  # 全局批次大小
        self.num_steps = args.num_steps  # 总训练步数
        self.num_epochs = self.num_steps // len(self.data) + 1  # 计算总训练轮数

        # 创建用于记录和评估的步骤列表
        self.chunks = np.reshape(np.array_split(range(self.num_steps), int(self.num_steps / 10))[10_000::10], (-1))

        # 检查是否可以使用CUDA进行同步
        self.sync_cuda = torch.cuda.is_available()

        # TensorBoard写入器
        self.writer = writer

        # 加载和同步模型参数
        self._load_and_sync_parameters()

        # 创建混合精度训练器
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=False,  # 不使用FP16
            fp16_scale_growth=1e-3  # FP16缩放增长率
        )
        # 设置保存目录和覆盖选项
        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        # 初始化优化器
        self.opt = AdamW(self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()

        # 设置设备
        self.device = torch.device(f"cuda:{self.rank}") if torch.cuda.is_available() else torch.device("cpu")

        # 创建采样器和设置评估相关变量
        self.schedule_sampler = create_named_schedule_sampler("uniform", diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        # 加载和同步模型参数
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"正在从检查点加载模型: {resume_checkpoint}...")
            self.model.load_state_dict(load_state_dict(resume_checkpoint, map_location=dev()))

    def _load_optimizer_state(self):
        # 加载优化器状态
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(bf.dirname(main_checkpoint), f"opt{self.resume_step:09d}.pt")
        if bf.exists(opt_checkpoint):
            logger.log(f"正在从检查点加载优化器状态: {opt_checkpoint}")
            self.opt.load_state_dict(load_state_dict(opt_checkpoint, map_location=dev()))

    def _print_stats(self):
        # 打印训练统计信息
        if (self.step % 100 == 0 and self.step > 0) and self.rank == 0:
            loss = logger.get_current().name2val["loss"]
            print(f"步骤[{self.step + self.resume_step}]: 损失[{loss:0.5f}]")

    def _write_to_logger(self):
        # 将训练信息写入日志
        if (self.step % self.log_interval == 0) and self.rank == 0:
            for k, v in logger.get_current().name2val.items():
                if k == "loss":
                    print(f"步骤[{self.step + self.resume_step}]: 损失[{v:0.5f}]")
                    self.writer.add_scalar(f"./Train/{k}", v, self.step)
                elif k not in ["step", "samples"] and "_q" not in k:
                    self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name="Loss")
                    self.writer.add_scalar(f"./Train/{k}", v, self.step)

    def run_loop(self):
        # 运行训练循环
        for _ in range(self.num_epochs):
            if self.rank == 0:
                try:
                    prof = profile.Profile()
                    prof.enable()
                except ValueError as e:
                    print(f"警告：无法启用 profiler。{str(e)}")

            for motion, cond in tqdm(self.data, disable=(self.rank != 0)):
                if self.lr_anneal_steps and self.step + self.resume_step >= self.lr_anneal_steps:
                    break

                motion = motion.to(self.device)
                cond["y"] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond["y"].items()}
                self.run_step(motion, cond)
                self._print_stats()
                self._write_to_logger()
                if (self.step % self.save_interval == 0) and self.rank == 0:
                    self.save()

                self.step += 1

                if (self.step == 1000) and self.rank == 0:
                    try:
                        prof.disable()
                        stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
                        stats.print_stats(10)
                    except Exception as e:
                        print(f"警告：无法禁用或打印 profiler 统计信息。{str(e)}")
            if self.lr_anneal_steps and self.step + self.resume_step >= self.lr_anneal_steps:
                break

        if ((self.step - 1) % self.save_interval != 0) and self.rank == 0:
            self.save()

    def run_step(self, batch: torch.Tensor, cond: Dict[str, Any]):
        # 运行单个训练步骤
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        if self.rank == 0:
            self.log_step()



    def forward_backward(self, batch: torch.Tensor, cond: Dict[str, Any]):
        # 前向传播和反向传播
        self.mp_trainer.zero_grad()
        t, weights = self.schedule_sampler.sample(batch.shape[0], batch.device)

        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model,
            batch,
            t,
            model_kwargs=cond,
        )

        losses = compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

        loss = (losses["loss"] * weights).mean()
        log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
        self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        # 学习率退火
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        # 记录步骤信息
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def ckpt_file_name(self):
        # 生成检查点文件名
        return f"model{(self.step+self.resume_step):09d}.pt"

    def save(self):
        # 保存模型和优化器状态
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("clip_model.")}
            logger.log("正在保存模型...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"), "wb") as f:
            torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename: str) -> int:
    """
    从形如 path/to/model.pt 的文件名中解析步骤数，其中  是检查点的步骤数。
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # 获取日志目录
    return logger.get_dir()


def find_resume_checkpoint():
    # 查找恢复检查点
    return None


def log_loss_dict(diffusion: Any, ts: torch.Tensor, losses: Dict[str, torch.Tensor]):
    # 记录损失字典
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

