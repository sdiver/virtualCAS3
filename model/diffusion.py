"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import json
from typing import Callable, Optional

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from model.guide import GuideTransformer
from model.modules.audio_encoder import Wav2VecEncoder
from model.modules.rotary_embedding_torch import RotaryEmbedding
from model.modules.transformer_modules import (
    DecoderLayerStack,
    FiLMTransformerDecoderLayer,
    RegressionTransformer,
    TransformerEncoderLayerRotary,
)
from model.utils import (
    init_weight,
    PositionalEncoding,
    prob_mask_like,
    setup_lip_regressor,
    SinusoidalPosEmb,
)
from model.vqvae import setup_tokenizer
from torch.nn import functional as F
from utils.misc import prGreen, prRed


class Audio2LipRegressionTransformer(torch.nn.Module):
    def __init__(
        self,
        n_vertices: int = 338,
        causal: bool = False,
        train_wav2vec: bool = False,
        transformer_encoder_layers: int = 2,
        transformer_decoder_layers: int = 4,
    ):
        super().__init__()
        self.n_vertices = n_vertices

        # 音频编码器
        self.audio_encoder = Wav2VecEncoder()
        if not train_wav2vec:
            self.audio_encoder.eval()
            for param in self.audio_encoder.parameters():
                param.requires_grad = False

        # 回归模型
        self.regression_model = RegressionTransformer(
            transformer_encoder_layers=transformer_encoder_layers,
            transformer_decoder_layers=transformer_decoder_layers,
            d_model=512,
            d_cond=512,
            num_heads=4,
            causal=causal,
        )
        # 输出投影层
        self.project_output = torch.nn.Linear(512, self.n_vertices * 3)

    def forward(self, audio):
        """
        :param audio: tensor of shape B x T x 1600
        :return: tensor of shape B x T x n_vertices x 3 containing reconstructed lip geometry
        """
        B, T = audio.shape[0], audio.shape[1]

        # 编码音频
        cond = self.audio_encoder(audio)

        # 生成初始输入
        x = torch.zeros(B, T, 512, device=audio.device)
        # 通过回归模型
        x = self.regression_model(x, cond)
        # 投影输出
        x = self.project_output(x)

        # 重塑输出为顶点坐标
        verts = x.view(B, T, self.n_vertices, 3)
        return verts


class FiLMTransformer(nn.Module):
    def __init__(
        self,
        args,
        nfeats: int,
        latent_dim: int = 512,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_feature_dim: int = 4800,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
        use_rotary: bool = True,
        cond_mode: str = "audio",
        split_type: str = "train",
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__()

        # 新增：情感编码器
        self.emotion_encoder = nn.Sequential(
            nn.Linear(args.emotion_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        # 更新条件特征维度以包含情感
        self.cond_feature_dim = cond_feature_dim + latent_dim

        self.nfeats = nfeats
        self.cond_mode = cond_mode
        # self.cond_feature_dim = cond_feature_dim
        self.add_frame_cond = args.add_frame_cond
        self.data_format = args.data_format
        self.split_type = split_type
        self.device = device

        # 位置编码
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        # 如果使用旋转位置编码，替换绝对位置编码
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )

        # 时间嵌入处理
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )
        self.to_time_cond = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim),
        )
        self.to_time_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),
            Rearrange("b (r d) -> b r d", r=2),
        )

        # 用于指导dropout的空嵌入
        self.seq_len = args.max_seq_length
        emb_len = 1998  # 暂时硬编码
        self.null_cond_embed = nn.Parameter(torch.randn(1, emb_len, latent_dim))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))
        self.norm_cond = nn.LayerNorm(latent_dim)
        self.setup_audio_models()

        # 设置姿势/面部特定的模型部分
        self.input_projection = nn.Linear(self.nfeats, latent_dim)
        if self.data_format == "pose":
            cond_feature_dim = 1024
            key_feature_dim = 104
            self.step = 30
            self.use_cm = True
            self.setup_guide_models(args, latent_dim, key_feature_dim)
            self.post_pose_layers = self._build_single_pose_conv(self.nfeats)
            self.post_pose_layers.apply(init_weight)
            self.final_conv = torch.nn.Conv1d(self.nfeats, self.nfeats, kernel_size=1)
            self.receptive_field = 25
        elif self.data_format == "face":
            self.use_cm = False
            cond_feature_dim = 1024 + 1014
            self.setup_lip_models()
            self.cond_encoder = nn.Sequential()
            for _ in range(2):
                self.cond_encoder.append(
                    TransformerEncoderLayerRotary(
                        d_model=latent_dim,
                        nhead=num_heads,
                        dim_feedforward=ff_size,
                        dropout=dropout,
                        activation=activation,
                        batch_first=True,
                        rotary=self.rotary,
                    )
                )
            self.cond_encoder.apply(init_weight)

        # 更新条件投影以适应新的维度
        self.cond_projection = nn.Linear(self.cond_feature_dim, latent_dim)
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # 解码器
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                    use_cm=self.use_cm,
                )
            )
        self.seqTransDecoder = DecoderLayerStack(decoderstack)
        self.seqTransDecoder.apply(init_weight)
        self.final_layer = nn.Linear(latent_dim, self.nfeats)
        self.final_layer.apply(init_weight)

    def _build_single_pose_conv(self, nfeats: int) -> nn.ModuleList:
        # 构建单个姿势的卷积层
        post_pose_layers = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(nfeats, max(256, nfeats), kernel_size=3, dilation=1),
                torch.nn.Conv1d(max(256, nfeats), nfeats, kernel_size=3, dilation=2),
                torch.nn.Conv1d(nfeats, nfeats, kernel_size=3, dilation=3),
                torch.nn.Conv1d(nfeats, nfeats, kernel_size=3, dilation=1),
                torch.nn.Conv1d(nfeats, nfeats, kernel_size=3, dilation=2),
                torch.nn.Conv1d(nfeats, nfeats, kernel_size=3, dilation=3),
            ]
        )
        return post_pose_layers

    def _run_single_pose_conv(self, output: torch.Tensor) -> torch.Tensor:
        # 运行单个姿势的卷积层
        output = torch.nn.functional.pad(output, pad=[self.receptive_field - 1, 0])
        for _, layer in enumerate(self.post_pose_layers):
            y = torch.nn.functional.leaky_relu(layer(output), negative_slope=0.2)
            if self.split_type == "train":
                y = torch.nn.functional.dropout(y, 0.2)
            if output.shape[1] == y.shape[1]:
                output = (output[:, :, -y.shape[-1] :] + y) / 2.0  # 跳跃连接
            else:
                output = y
        return output

    def setup_guide_models(self, args, latent_dim: int, key_feature_dim: int) -> None:
        # 设置指导模型
        max_keyframe_len = len(list(range(self.seq_len))[:: self.step])
        self.null_pose_embed = nn.Parameter(
            torch.randn(1, max_keyframe_len, latent_dim)
        )
        prGreen(f"using keyframes: {self.null_pose_embed.shape}")
        self.frame_cond_projection = nn.Linear(key_feature_dim, latent_dim)
        self.frame_norm_cond = nn.LayerNorm(latent_dim)
        # 为测试时设置关键帧转换器
        self.resume_trans = None
        if self.split_type == "test":
            if hasattr(args, "resume_trans") and args.resume_trans is not None:
                self.resume_trans = args.resume_trans
                self.setup_guide_predictor(args.resume_trans)
            else:
                prRed("not using transformer, just using ground truth")

    def setup_guide_predictor(self, cp_path: str) -> None:
        # 设置指导预测器
        cp_dir = cp_path.split("checkpoints/iter-")[0]
        with open(f"{cp_dir}/args.json") as f:
            trans_args = json.load(f)

        # 根据转换参数加载点设置标记器
        self.tokenizer = setup_tokenizer(trans_args["resume_pth"])

        # 设置转换器
        self.transformer = GuideTransformer(
            tokens=self.tokenizer.n_clusters,
            num_layers=trans_args["layers"],
            dim=trans_args["dim"],
            emb_len=1998,
            num_audio_layers=trans_args["num_audio_layers"],
        )
        for param in self.transformer.parameters():
            param.requires_grad = False
        prGreen("loading TRANSFORMER checkpoint from {}".format(cp_path))
        cp = torch.load(cp_path)
        missing_keys, unexpected_keys = self.transformer.load_state_dict(
            cp["model_state_dict"], strict=False
        )
        assert len(missing_keys) == 0, missing_keys
        assert len(unexpected_keys) == 0, unexpected_keys

    def setup_audio_models(self) -> None:
        # 设置音频模型
        self.audio_model, self.audio_resampler = setup_lip_regressor()

    def setup_lip_models(self) -> None:
        # 设置唇部模型
        self.lip_model = Audio2LipRegressionTransformer()
        cp_path = "./assets/iter-0200000.pt"
        cp = torch.load(cp_path, map_location=torch.device(self.device))
        self.lip_model.load_state_dict(cp["model_state_dict"])
        for param in self.lip_model.parameters():
            param.requires_grad = False
        prGreen(f"adding lip conditioning {cp_path}")

    def parameters_w_grad(self):
        # 返回具有梯度的参数
        return [p for p in self.parameters() if p.requires_grad]

    def encode_audio(self, raw_audio: torch.Tensor, emotion: torch.Tensor) -> torch.Tensor:
        # 编码音频
        device = next(self.parameters()).device
        a0 = self.audio_resampler(raw_audio[:, :, 0].to(device))
        a1 = self.audio_resampler(raw_audio[:, :, 1].to(device))
        with torch.no_grad():
            z0 = self.audio_model.feature_extractor(a0)
            z1 = self.audio_model.feature_extractor(a1)
            emb = torch.cat((z0, z1), axis=1).permute(0, 2, 1)
        # 编码情感
        emotion_embedding = self.emotion_encoder(emotion)
        # 将情感嵌入添加到音频嵌入中
        emb = torch.cat((emb, emotion_embedding.unsqueeze(1).repeat(1, emb.shape[1], 1)), dim=-1)

        return emb

    def encode_lip(self, audio: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        # 编码唇部
        reshaped_audio = audio.reshape((audio.shape[0], -1, 1600, 2))[..., 0]
        # 每次处理4秒
        B, T, _ = reshaped_audio.shape
        lip_cond = torch.zeros(
            (audio.shape[0], T, 338, 3),
            device=audio.device,
            dtype=audio.dtype,
        )
        for i in range(0, T, 120):
            lip_cond[:, i : i + 120, ...] = self.lip_model(
                reshaped_audio[:, i : i + 120, ...]
            )
        lip_cond = lip_cond.permute(0, 2, 3, 1).reshape((B, 338 * 3, -1))
        lip_cond = torch.nn.functional.interpolate(
            lip_cond, size=cond_embed.shape[1], mode="nearest-exact"
        ).permute(0, 2, 1)
        cond_embed = torch.cat((cond_embed, lip_cond), dim=-1)
        return cond_embed

    def encode_keyframes(
        self, y: torch.Tensor, cond_drop_prob: float, batch_size: int
    ) -> torch.Tensor:
        pred = y["keyframes"]
        new_mask = y["mask"][..., :: self.step].squeeze((1, 2))
        pred[~new_mask] = 0.0  # pad the unknown
        pose_hidden = self.frame_cond_projection(pred.detach().clone().cuda())
        pose_embed = self.abs_pos_encoding(pose_hidden)
        pose_tokens = self.frame_norm_cond(pose_embed)
        # do conditional dropout for guide poses
        key_cond_drop_prob = cond_drop_prob
        keep_mask_pose = prob_mask_like(
            (batch_size,), 1 - key_cond_drop_prob, device=pose_tokens.device
        )
        keep_mask_pose_embed = rearrange(keep_mask_pose, "b -> b 1 1")
        null_pose_embed = self.null_pose_embed.to(pose_tokens.dtype)

        # # 对齐null cond embed 的维度
        #
        # if self.null_pose_embed.shape[1] < pose_tokens.shape[1]:
        #     padding_size = pose_tokens.shape[1] - self.null_pose_embed.shape[1]
        #     # 通过填充零的方式扩展 null_cond_embed
        #     null_pose_embed = torch.cat([
        #         self.null_cond_embed,
        #         torch.zeros(1, padding_size, self.null_cond_embed.shape[2]).to(self.null_cond_embed.device)
        #     ], dim=1)
        #
        # null_pose_embed = null_pose_embed.to(pose_tokens.dtype)

        pose_tokens = torch.where(
            keep_mask_pose_embed,
            pose_tokens,
            null_pose_embed[:, : pose_tokens.shape[1], :],
        )
        return pose_tokens

    def forward(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        cond_drop_prob: float = 0.0,
    ) -> torch.Tensor:
        if x.dim() == 4:
            x = x.permute(0, 3, 1, 2).squeeze(-1)
        batch_size, device = x.shape[0], x.device
        if self.cond_mode == "uncond":
            cond_embed = torch.zeros(
                (x.shape[0], x.shape[1], self.cond_feature_dim),
                dtype=x.dtype,
                device=x.device,
            )
        else:
            cond_embed = y["audio"]
            emotion = y["emotion"]
            # 假设情感标签在 y 字典中
            cond_embed = self.encode_audio(cond_embed, emotion)
            if self.data_format == "face":
                cond_embed = self.encode_lip(y["audio"], cond_embed)
                pose_tokens = None
            if self.data_format == "pose":
                pose_tokens = self.encode_keyframes(y, cond_drop_prob, batch_size)
        assert cond_embed is not None, "cond emb should not be none"
        # process conditioning information
        x = self.input_projection(x)
        x = self.abs_pos_encoding(x)
        audio_cond_drop_prob = cond_drop_prob
        keep_mask = prob_mask_like(
            (batch_size,), 1 - audio_cond_drop_prob, device=device
        )
        keep_mask_embed = rearrange(keep_mask, "b -> b 1 1")
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1")
        cond_tokens = self.cond_projection(cond_embed)
        cond_tokens = self.abs_pos_encoding(cond_tokens)
        if self.data_format == "face":
            cond_tokens = self.cond_encoder(cond_tokens)
        null_cond_embed = self.null_cond_embed.to(cond_tokens.dtype)
        # # 填充或复制 null_cond_embed 使其匹配 cond_tokens 的长度
        # if self.null_cond_embed.shape[1] < cond_tokens.shape[1]:
        #     padding_size = cond_tokens.shape[1] - self.null_cond_embed.shape[1]
        #     # 通过填充零的方式扩展 null_cond_embed
        #     null_cond_embed = torch.cat([
        #         self.null_cond_embed,
        #         torch.zeros(1, padding_size, self.null_cond_embed.shape[2]).to(self.null_cond_embed.device)
        #     ], dim=1)
        #
        # null_cond_embed = null_cond_embed.to(cond_tokens.dtype)
        # # 打印 cond_tokens 及其形状
        # print("cond_tokens shape:", cond_tokens.shape)
        #
        # # 打印 keep_mask_embed 及其形状
        # print("keep_mask_embed shape:", keep_mask_embed.shape)
        #
        # # 打印 null_cond_embed 的原始形状及切片后的形状
        # print("null_cond_embed original shape:", null_cond_embed.shape)
        # print("null_cond_embed[:, : cond_tokens.shape[1], :] shape:",
        #       null_cond_embed[:, : cond_tokens.shape[1], :].shape)
        #
        # # 打印 keep_mask_embed 是否与 cond_tokens 和 null_cond_embed 匹配
        # keep_mask_embed_shape = keep_mask_embed.shape
        # cond_tokens_shape = cond_tokens.shape
        # null_cond_embed_shape = null_cond_embed[:, : cond_tokens.shape[1], :].shape
        #
        # # 检查所有张量在 torch.where 中的兼容性
        # print("keep_mask_embed shape for torch.where:", keep_mask_embed_shape)
        # print("cond_tokens shape for torch.where:", cond_tokens_shape)
        # print("null_cond_embed shape for torch.where:", null_cond_embed_shape)

        cond_tokens = torch.where(
            keep_mask_embed, cond_tokens, null_cond_embed[:, : cond_tokens.shape[1], :]
        )
        mean_pooled_cond_tokens = cond_tokens.mean(dim=-2)
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens)

        # create t conditioning
        t_hidden = self.time_mlp(times)
        t = self.to_time_cond(t_hidden)
        t_tokens = self.to_time_tokens(t_hidden)
        null_cond_hidden = self.null_cond_hidden.to(t.dtype)
        cond_hidden = torch.where(keep_mask_hidden, cond_hidden, null_cond_hidden)
        t += cond_hidden

        # cross-attention conditioning
        c = torch.cat((cond_tokens, t_tokens), dim=-2)
        cond_tokens = self.norm_cond(c)

        # Pass through the transformer decoder
        output = self.seqTransDecoder(x, cond_tokens, t, memory2=pose_tokens)
        output = self.final_layer(output)
        if self.data_format == "pose":
            output = output.permute(0, 2, 1)
            output = self._run_single_pose_conv(output)
            output = self.final_conv(output)
            output = output.permute(0, 2, 1)
        return output
