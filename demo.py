"""
版权所有 (c) Meta Platforms, Inc. 及其附属公司。
保留所有权利。
本源代码根据本源代码树根目录中的 LICENSE 文件中的许可证授权。
"""

# 导入必要的库
import json
from typing import Dict, Union

import gradio as gr
import numpy as np
import torch
import torchaudio
from attrdict import AttrDict


from diffusion.respace import SpacedDiffusion
from model.cfg_sampler import ClassifierFreeSampleModel
from model.diffusion import FiLMTransformer
from utils.misc import fix_seed
from utils.model_util import create_model_and_diffusion, load_model
from visualize.render_codes import BodyRenderer


class GradioModel:
    def __init__(self, face_args, pose_args) -> None:
        # 初始化面部模型和姿势模型
        self.face_model, self.face_diffusion, self.device = self._setup_model(
            # face_args, "checkpoints-bak/checkpoints-184/diffusion/c1_face/model000155000.pt"
            face_args, "checkpoints/diffusion/c1_face_secondtest/model000000000.pt"
        )

        print("--------------------------------------------pose exxxx")
        self.pose_model, self.pose_diffusion, _ = self._setup_model(
            # pose_args, "checkpoints-bak/checkpoints-184/diffusion/c1_pose/model000340000.pt"
            pose_args, "checkpoints/diffusion/c1_pose_test_123/model000811680.pt"
        )
        # 加载标准化数据
        stats = torch.load("dataset/PXB184/data_stats.pth")
        stats["pose_mean"] = stats["pose_mean"].reshape(-1)
        stats["pose_std"] = stats["pose_std"].reshape(-1)
        print("Stats overview:", stats)

        for key, value in stats.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}:")
                print(f"  Shape: {value.shape}")
                print(f"  Data type: {value.dtype}")
                print(f"  Min: {value.min():.5f}")
                print(f"  Max: {value.max():.5f}")
                print(f"  Mean: {value.mean():.5f}")
                print(f"  Std: {value.std():.5f}")
            else:
                print(f"{key}: {value}")
        self.stats = stats
        # 设置渲染器
        config_base = f"./checkpoints-bak/checkpoints-184/PXB184"
        self.body_renderer = BodyRenderer(
            config_base=config_base,
            render_rgb=True,
        )

    def _setup_model(
        self,
        args_path: str,
        model_path: str,
    ) -> (Union[FiLMTransformer, ClassifierFreeSampleModel], SpacedDiffusion):
        """
        设置模型的辅助函数

        参数:
        args_path: 模型参数文件路径
        model_path: 模型权重文件路径

        返回:
        model: 设置好的模型
        diffusion: 扩散模型
        device: 运行设备
        """
        with open(args_path) as f:
            args = json.load(f)
        args = AttrDict(args)
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("running on...", args.device)
        args.model_path = model_path
        args.output_dir = "/tmp/gradio/"
        args.timestep_respacing = "ddim100"
        if args.data_format == "pose":
            args.resume_trans = "checkpoints-bak/checkpoints-184/guide/c1_pose/checkpoints/iter-0100000.pt"
            # args.resume_trans = "checkpoints/guide/c1_pose/checkpoints/iter-0100000.pt"

        # 创建模型
        model, diffusion = create_model_and_diffusion(args, split_type="test")
        print(f"Loading checkpoints from [{args.model_path}]...")
        state_dict = torch.load(args.model_path, map_location=args.device)
        load_model(model, state_dict)
        model = ClassifierFreeSampleModel(model)
        model.eval()
        model.to(args.device)
        return model, diffusion, args.device

    def _replace_keyframes(
        self,
        model_kwargs: Dict[str, Dict[str, torch.Tensor]],
        B: int,
        T: int,
        top_p: float = 0.97,
    ) -> torch.Tensor:
        """
        替换关键帧的辅助函数

        参数:
        model_kwargs: 模型输入参数
        B: batch大小
        T: 序列长度
        top_p: 采样多样性参数

        返回:
        pred: 预测的关键帧
        """
        with torch.no_grad():
            tokens = self.pose_model.transformer.generate(
                model_kwargs["y"]["audio"],
                T,
                layers=self.pose_model.tokenizer.residual_depth,
                n_sequences=B,
                top_p=top_p,
            )
        tokens = tokens.reshape((B, -1, self.pose_model.tokenizer.residual_depth))
        pred = self.pose_model.tokenizer.decode(tokens).detach()
        return pred

    def _run_single_diffusion(
        self,
        model_kwargs: Dict[str, Dict[str, torch.Tensor]],
        diffusion: SpacedDiffusion,
        model: Union[FiLMTransformer, ClassifierFreeSampleModel],
        curr_seq_length: int,
        num_repetitions: int = 1,
    ) -> (torch.Tensor,):
        """
        运行单个扩散过程的辅助函数

        参数:
        model_kwargs: 模型输入参数
        diffusion: 扩散模型
        model: 生成模型
        curr_seq_length: 当前序列长度
        num_repetitions: 重复次数

        返回:
        sample: 生成的样本
        """
        sample_fn = diffusion.ddim_sample_loop
        with torch.no_grad():
            sample = sample_fn(
                model,
                (num_repetitions, model.nfeats, 1, curr_seq_length),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
        return sample

    def generate_sequences(
        self,
        model_kwargs: Dict[str, Dict[str, torch.Tensor]],
        data_format: str,
        curr_seq_length: int,
        num_repetitions: int = 5,
        guidance_param: float = 10.0,
        top_p: float = 0.97,
    ) -> Dict[str, np.ndarray]:
        """
        生成序列的主要函数

        参数:
        model_kwargs: 模型输入参数
        data_format: 数据格式 ('pose' 或 'face')
        curr_seq_length: 当前序列长度
        num_repetitions: 重复次数
        guidance_param: 引导参数
        top_p: 采样多样性参数

        返回:
        生成的序列
        """
        if data_format == "pose":
            model = self.pose_model
            diffusion = self.pose_diffusion
        else:
            model = self.face_model
            diffusion = self.face_diffusion

                # 初始化存储所有生成动作的列表
        all_motions = []
        # 设置引导参数
        model_kwargs["y"]["scale"] = torch.ones(num_repetitions) * guidance_param
        # 将模型输入参数移到指定设备上
        model_kwargs["y"] = {
            key: val.to(self.device) if torch.is_tensor(val) else val
            for key, val in model_kwargs["y"].items()
        }
        # 如果是姿势数据，设置掩码和关键帧
        if data_format == "pose":
            model_kwargs["y"]["mask"] = (
                torch.ones((num_repetitions, 1, 1, curr_seq_length))
                .to(self.device)
                .bool()
            )
            model_kwargs["y"]["keyframes"] = self._replace_keyframes(
                model_kwargs,
                num_repetitions,
                int(curr_seq_length / 30),
                top_p=top_p,
            )
        # 运行扩散过程生成样本
        sample = self._run_single_diffusion(
            model_kwargs, diffusion, model, curr_seq_length, num_repetitions
        )
        # 将生成的样本添加到列表中
        all_motions.append(sample.cpu().numpy())
        print(f"created {len(all_motions) * num_repetitions} samples")
        # 返回连接后的所有动作
        return np.concatenate(all_motions, axis=0)


def generate_results(audio: np.ndarray, num_repetitions: int, top_p: float):
    """
    生成结果的函数

    参数:
    audio: 输入的音频数据
    num_repetitions: 生成样本的数量
    top_p: 采样多样性参数

    返回:
    face_results: 生成的面部动作
    pose_results: 生成的姿势动作
    dual_audio: 处理后的音频数据
    """
    # 检查是否有输入音频
    if audio is None:
        raise gr.Error("Please record audio to start")
    sr, y = audio
    # 将音频转换为单声道并进行重采样
    y = torch.Tensor(y)
    if y.dim() == 2:
        dim = 0 if y.shape[0] == 2 else 1
        y = torch.mean(y, dim=dim)
    y = torchaudio.functional.resample(torch.Tensor(y), orig_freq=sr, new_freq=48_000)
    sr = 48_000
    # 确保音频长度至少为4秒
    if len(y) < (sr * 4):
        raise gr.Error("Please record at least 4 second of audio")
    # 验证生成样本的数量
    if num_repetitions is None or num_repetitions <= 0 or num_repetitions > 10:
        raise gr.Error(
            f"Invalid number of samples: {num_repetitions}. Please specify a number between 1-10"
        )
    # 调整音频长度
    cutoff = int(len(y) / (sr * 4))
    y = y[: cutoff * sr * 4]
    curr_seq_length = int(len(y) / sr) * 30
    # 创建模型输入参数
    model_kwargs = {"y": {}}
    # dual_audio = np.random.normal(0, 0.001, (1, len(y), 2))
    # # dual_audio[:, :, 0] = y.copy_() / max(y)
    # dual_audio[:, :, 0] = torch.from_numpy(y) / torch.max(torch.abs(torch.from_numpy(y)))
    # # 标准化音频数据
    # dual_audio = (dual_audio - gradio_model.stats["audio_mean"]) / gradio_model.stats[
    #     "audio_std_flat"
    # ]
    dual_audio = np.random.normal(0, 0.001, (1, len(y), 2))
    dual_audio[:, :, 0] = y.detach().cpu().numpy() / np.max(np.abs(y.detach().cpu().numpy()))
    # Normalize audio data
    dual_audio = (dual_audio - gradio_model.stats["audio_mean"]) / gradio_model.stats[
        "audio_std_flat"
    ]
    model_kwargs["y"]["audio"] = (
        torch.Tensor(dual_audio).float().tile(num_repetitions, 1, 1)
    )
    # 生成面部动作结果
    face_results = (
        gradio_model.generate_sequences(
            model_kwargs, "face", curr_seq_length, num_repetitions=int(num_repetitions)
        )
        .squeeze(2)
        .transpose(0, 2, 1)
    )
    face_results = (
        face_results * gradio_model.stats["code_std"] + gradio_model.stats["code_mean"]
    )
    # 生成姿势动作结果
    pose_results = (
        gradio_model.generate_sequences(
            model_kwargs,
            "pose",
            curr_seq_length,
            num_repetitions=int(num_repetitions),
            guidance_param=2.0,
            top_p=top_p,
        )
        .squeeze(2)
        .transpose(0, 2, 1)
    )
    pose_results = (
        pose_results * gradio_model.stats["pose_std"] + gradio_model.stats["pose_mean"]
    )
    # 反标准化音频数据
    dual_audio = (
        dual_audio * gradio_model.stats["audio_std_flat"]
        + gradio_model.stats["audio_mean"]
    )
    return face_results, pose_results, dual_audio[0].transpose(1, 0).astype(np.float32)


def audio_to_avatar(audio: np.ndarray, num_repetitions: int, top_p: float):
    """
    将音频转换为头像的主函数

    参数:
    audio: 输入的音频数据
    num_repetitions: 生成样本的数量
    top_p: 采样多样性参数

    返回:
    results: 生成的视频列表
    """
    # 生成面部和姿势结果
    face_results, pose_results, audio = generate_results(audio, num_repetitions, top_p)
    # 返回: num_rep x T x 104
    B = len(face_results)
    results = []
    # 为每个生成的样本渲染视频
    for i in range(B):
        render_data_block = {
            "audio": audio,  # 2 x T
            "body_motion": pose_results[i, ...],  # T x 104
            "face_motion": face_results[i, ...],  # T x 256
        }
        gradio_model.body_renderer.render_full_video(
            render_data_block, f"/tmp/sample{i}", audio_sr=48_000
        )
        results += [gr.Video(value=f"/tmp/sample{i}_pred.mp4", visible=True)]
    # 添加空的视频占位符
    results += [gr.Video(visible=False) for _ in range(B, 10)]
    return results


# 初始化Gradio模型
gradio_model = GradioModel(
    # face_args="./checkpoints-bak/checkpoints-184/diffusion/c1_face/args.json",
    face_args="./checkpoints/diffusion/c1_face_secondtest/args.json",
    # pose_args="./checkpoints-bak/checkpoints-184/diffusion/c1_pose/args.json",
    pose_args="./checkpoints/diffusion/c1_pose_test_123/args.json",
)

# 创建Gradio界面
demo = gr.Interface(
    audio_to_avatar,  # 主函数
    [
        gr.Audio(sources=["microphone","upload"]),
        gr.Number(
            value=1,
            label="Number of Samples (default = 1)",
            precision=0,
            minimum=1,
            maximum=10,
        ),
        gr.Number(
            value=0.97,
            label="Sample Diversity (default = 0.97)",
            precision=None,
            minimum=0.01,
            step=0.01,
            maximum=1.00,
        ),
    ],  # 输入类型
    [gr.Video(format="mp4", visible=True)]
    + [gr.Video(format="mp4", visible=False) for _ in range(9)],  # 输出类型
    title='"从音频到真实感化身:在对话中合成人类"演示',
    description="您可以从您的声音生成逼真的头像! <br/>\
        1) 首先录制您的音频。 <br/>\
        2) 指定要生成的样本数量。 <br/>\
        3) 指定您希望样本的多样性程度。这会调整核采样中的累积概率:0.01 = 低多样性,1.0 = 高多样性。 <br/>\
        4) 然后,坐下来等待渲染完成!这可能需要一段时间(例如30分钟) <br/>\
        5) 之后,您可以查看视频并下载您喜欢的视频。 <br/>",
    article="相关链接: [项目页面](https://www.humanplus.xyz)",  # TODO: 代码和arxiv
)

if __name__ == "__main__":
    fix_seed(10)  # 设置随机种子
    demo.launch(share=True)  # 启动Gradio演示
