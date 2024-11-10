import copy
import json
from typing import Dict, Union

import gradio as gr
import numpy as np
import torch
import torchaudio
from attrdict import AttrDict
from torch.utils.data import DataLoader
from torch.quantization import quantize_dynamic
from typing import Dict, Union, Tuple

from diffusion.respace import SpacedDiffusion
from model.cfg_sampler import ClassifierFreeSampleModel
from model.diffusion import FiLMTransformer
from utils.misc import fix_seed
from utils.model_util import create_model_and_diffusion, load_model
from visualize.render_codes import BodyRenderer


class GradioModel:
    def __init__(self, face_args, pose_args) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_model, self.face_diffusion = self._setup_model(
            face_args, "checkpoints-bak/checkpoints-184/diffusion/c1_face/model000155000.pt"
        )
        self.pose_model, self.pose_diffusion = self._setup_model(
            pose_args, "checkpoints-bak/checkpoints-184/diffusion/c1_pose/model000340000.pt"
        )

        # Load stats and move to the correct device
        # stats = torch.load("dataset/PXB184/data_stats.pth", map_location=self.device)
        # stats["pose_mean"] = stats["pose_mean"].reshape(-1).to(self.device)
        # stats["pose_std"] = stats["pose_std"].reshape(-1).to(self.device)
        # stats["audio_mean"] = stats["audio_mean"].to(self.device)
        # stats["audio_std_flat"] = stats["audio_std_flat"].to(self.device)
        # stats["code_mean"] = stats["code_mean"].to(self.device)
        # stats["code_std"] = stats["code_std"].to(self.device)
        # self.stats = stats

        config_base = f"./checkpoints-bak/checkpoints-184/PXB184"
        self.body_renderer = BodyRenderer(config_base=config_base, render_rgb=True)

    def _setup_model(self, args_path: str, model_path: str) -> (
    Union[FiLMTransformer, ClassifierFreeSampleModel], SpacedDiffusion):
        with open(args_path) as f:
            args = AttrDict(json.load(f))
        args.device = self.device
        args.model_path = model_path
        args.output_dir = "/tmp/gradio/"
        args.timestep_respacing = "ddim100"
        if args.data_format == "pose":
            args.resume_trans = "checkpoints-bak/checkpoints-184/guide/c1_pose/checkpoints/iter-0100000.pt"

        model, diffusion = create_model_and_diffusion(args, split_type="test")
        print(f"Loading checkpoints from [{args.model_path}]...")
        state_dict = torch.load(args.model_path, map_location=self.device)
        load_model(model, state_dict)
        model = ClassifierFreeSampleModel(model)
        model.eval()
        model.to(self.device)

        # Apply dynamic quantization
        model = quantize_dynamic(model, {torch.nn.Linear})

        # Use JIT compilation
        # model = torch.jit.script(model)
        # Load stats and move to the correct device
        stats = torch.load("dataset/PXB184/data_stats.pth", map_location=self.device)
        self.stats = {
            "pose_mean": torch.from_numpy(stats["pose_mean"]).reshape(-1).to(self.device),
            "pose_std": torch.from_numpy(stats["pose_std"]).reshape(-1).to(self.device),
            "audio_mean": torch.from_numpy(stats["audio_mean"]).to(self.device),
            "audio_std_flat": torch.from_numpy(stats["audio_std_flat"]).to(self.device),
            "code_mean": torch.from_numpy(stats["code_mean"]).to(self.device),
            "code_std": torch.from_numpy(stats["code_std"]).to(self.device)
        }

        return model, diffusion

    @torch.no_grad()
    def _replace_keyframes(self, model_kwargs: Dict[str, Dict[str, torch.Tensor]], B: int, T: int,
                           top_p: float = 0.97) -> torch.Tensor:
        tokens = self.pose_model.transformer.generate(
            model_kwargs["y"]["audio"],
            T,
            layers=self.pose_model.tokenizer.residual_depth,
            n_sequences=B,
            top_p=top_p,
        )
        tokens = tokens.reshape((B, -1, self.pose_model.tokenizer.residual_depth))
        pred = self.pose_model.tokenizer.decode(tokens)
        return pred

    @torch.no_grad()
    def _run_single_diffusion(self, model_kwargs: Dict[str, Dict[str, torch.Tensor]], diffusion: SpacedDiffusion,
                              model: Union[FiLMTransformer, ClassifierFreeSampleModel], curr_seq_length: int,
                              num_repetitions: int = 1) -> torch.Tensor:
        sample_fn = diffusion.ddim_sample_loop
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

    @torch.no_grad()
    def generate_sequences(self, model_kwargs: Dict[str, Dict[str, torch.Tensor]], data_format: str,
                           curr_seq_length: int, num_repetitions: int = 5, guidance_param: float = 10.0,
                           top_p: float = 0.97) -> Dict[str, np.ndarray]:
        model = self.pose_model if data_format == "pose" else self.face_model
        diffusion = self.pose_diffusion if data_format == "pose" else self.face_diffusion

        all_motions = []
        model_kwargs["y"]["scale"] = torch.ones(num_repetitions, device=self.device) * guidance_param
        model_kwargs["y"] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in
                             model_kwargs["y"].items()}

        if data_format == "pose":
            model_kwargs["y"]["mask"] = torch.ones((num_repetitions, 1, 1, curr_seq_length), device=self.device).bool()
            model_kwargs["y"]["keyframes"] = self._replace_keyframes(model_kwargs, num_repetitions,
                                                                     int(curr_seq_length / 30), top_p=top_p)

        # Implement batch processing
        batch_size = 5  # Adjust based on your GPU memory
        for i in range(0, num_repetitions, batch_size):
            batch_repetitions = min(batch_size, num_repetitions - i)
            batch_model_kwargs = {
                k: {kk: vv[i:i + batch_repetitions] if torch.is_tensor(vv) else vv for kk, vv in v.items()} for k, v in
                model_kwargs.items()}
            sample = self._run_single_diffusion(batch_model_kwargs, diffusion, model, curr_seq_length,
                                                batch_repetitions)
            all_motions.append(sample.cpu())

        return torch.cat(all_motions, dim=0).numpy()


def generate_results(audio: np.ndarray, num_repetitions: int, top_p: float):
    if audio is None:
        raise gr.Error("Please record audio to start")
    sr, y = audio

    # Convert to float32 and normalize
    y = torch.tensor(y, dtype=torch.float32, device=gradio_model.device)
    y = y / torch.max(torch.abs(y))

    if y.dim() == 2:
        y = torch.mean(y, dim=0 if y.shape[0] == 2 else 1)

    y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=48_000)
    sr = 48_000
    if len(y) < (sr * 4):
        raise gr.Error("Please record at least 4 second of audio")
    if num_repetitions is None or num_repetitions <= 0 or num_repetitions > 10:
        raise gr.Error(f"Invalid number of samples: {num_repetitions}. Please specify a number between 1-10")

    cutoff = int(len(y) / (sr * 4))
    y = y[: cutoff * sr * 4]
    curr_seq_length = int(len(y) / sr) * 30
    model_kwargs = {"y": {}}
    dual_audio = torch.randn(1, len(y), 2, device=gradio_model.device) * 0.001
    dual_audio[:, :, 0] = y

    # Move stats to the same device as dual_audio
    audio_mean = gradio_model.stats["audio_mean"]
    audio_std_flat = gradio_model.stats["audio_std_flat"]

    dual_audio = (dual_audio - audio_mean) / audio_std_flat
    model_kwargs["y"]["audio"] = dual_audio.float().repeat(num_repetitions, 1, 1)

    face_results = gradio_model.generate_sequences(model_kwargs, "face", curr_seq_length,
                                                   num_repetitions=int(num_repetitions))
    face_results = face_results.squeeze(2).transpose(0, 2, 1)

    # Move stats to CPU for numpy operations
    code_std = gradio_model.stats["code_std"].cpu().numpy()
    code_mean = gradio_model.stats["code_mean"].cpu().numpy()
    face_results = face_results.cpu().numpy() * code_std + code_mean

    pose_results = gradio_model.generate_sequences(model_kwargs, "pose", curr_seq_length,
                                                   num_repetitions=int(num_repetitions), guidance_param=2.0,
                                                   top_p=top_p)
    pose_results = pose_results.squeeze(2).transpose(0, 2, 1)

    # Move stats to CPU for numpy operations
    pose_std = gradio_model.stats["pose_std"].cpu().numpy()
    pose_mean = gradio_model.stats["pose_mean"].cpu().numpy()
    pose_results = pose_results.cpu().numpy() * pose_std + pose_mean

    # Move dual_audio to CPU for final numpy operations
    dual_audio = dual_audio.cpu()
    dual_audio = dual_audio * audio_std_flat.cpu().numpy() + audio_mean.cpu().numpy()

    return face_results, pose_results, dual_audio[0].transpose(1, 0).numpy().astype(np.float32)


def audio_to_avatar(audio: np.ndarray, num_repetitions: int, top_p: float):
    face_results, pose_results, audio = generate_results(audio, num_repetitions, top_p)
    B = len(face_results)
    results = []

    for i in range(B):
        render_data_block = {
            "audio": audio,
            "body_motion": pose_results[i, ...],
            "face_motion": face_results[i, ...],
        }
        gradio_model.body_renderer.render_full_video(render_data_block, f"/tmp/sample{i}", audio_sr=48_000)
        results += [gr.Video(value=f"/tmp/sample{i}_pred.mp4", visible=True)]
    results += [gr.Video(visible=False) for _ in range(B, 10)]
    return results


gradio_model = GradioModel(
    face_args="./checkpoints-bak/checkpoints-184/diffusion/c1_face/args.json",
    pose_args="./checkpoints-bak/checkpoints-184/diffusion/c1_pose/args.json",
)

demo = gr.Interface(
    audio_to_avatar,
    [
        gr.Audio(sources=["microphone", "upload"]),
        gr.Number(value=1, label="Number of Samples (default = 1)", precision=0, minimum=1, maximum=10),
        gr.Number(value=0.97, label="Sample Diversity (default = 0.97)", precision=None, minimum=0.01, step=0.01,
                  maximum=1.00),
    ],
    [gr.Video(format="mp4", visible=True)] + [gr.Video(format="mp4", visible=False) for _ in range(9)],
    title='"From Audio to Realistic Avatar: Synthesizing Humans in Conversations" Demo',
    description="You can generate realistic avatars from your voice! <br/>\
        1) First, record your audio. <br/>\
        2) Specify the number of samples you want to generate. <br/>\
        3) Specify how diverse you want the samples to be. This adjusts the cumulative probability in nucleus sampling: 0.01 = low diversity, 1.0 = high diversity. <br/>\
        4) Then, sit back and wait for the rendering to complete! This may take a while (e.g., 30 minutes) <br/>\
        5) Afterwards, you can view the videos and download the ones you like. <br/>",
    article="Related links: [Project Page](https://www.humanplus.xyz)",
)

if __name__ == "__main__":
    fix_seed(10)
    demo.launch(share=True)