"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
from typing import Dict, Iterable, List, Union


import numpy as np
import torch
from torch.utils import data

from utils.misc import prGreen, prRed


class Social(data.Dataset):
    def __init__(
        self,
        args,
        data_dict: Dict[str, Iterable],
        split: str = "train",
        chunk: bool = False,
        add_padding: bool = True,
    ) -> None:
        if args.data_format == "face":
            prGreen("[dataset.py] training face only model")
            data_dict["data"] = data_dict["face"]
        elif args.data_format == "pose":
            prGreen("[dataset.py] training pose only model")
            missing = []
            for d in data_dict["data"]:
                missing.append(np.ones_like(d))
            data_dict["missing"] = missing
        # set up variables for dataloader
        self.data_format = args.data_format
        self.add_frame_cond = args.add_frame_cond
        self._register_keyframe_step()
        self.data_root = args.data_root
        self.max_seq_length = args.max_seq_length
        if hasattr(args, "curr_seq_length") and args.curr_seq_length is not None:
            self.max_seq_length = args.curr_seq_length
        prGreen([f"[dataset.py] sequences of {self.max_seq_length}"])
        self.add_padding = add_padding
        self.audio_per_frame = 1600
        self.max_audio_length = self.max_seq_length * self.audio_per_frame
        self.min_seq_length = 400
        # 添加情緒
        self.emotions = data_dict.get("emotions", None)

        # set up training/validation splits
        train_idx = list(range(0, len(data_dict["data"]) - 6))
        val_idx = list(range(len(data_dict["data"]) - 6, len(data_dict["data"]) - 4))
        test_idx = list(range(len(data_dict["data"]) - 4, len(data_dict["data"])))
        self.split = split
        if split == "train":
            self._pick_sequences(data_dict, train_idx)
        elif split == "val":
            self._pick_sequences(data_dict, val_idx)
        else:
            self._pick_sequences(data_dict, test_idx)
        self.chunk = chunk
        if split == "test":
            print("[dataset.py] chunking data...")
            self._chunk_data()
        self._load_std()
        prGreen(
            f"[dataset.py] {split} | {len(self.data)} sequences ({self.data[0].shape}) | total len {self.total_len}"
        )

    def inv_transform(
        self, data: Union[np.ndarray, torch.Tensor], data_type: str
    ) -> Union[np.ndarray, torch.Tensor]:
        if data_type == "pose":
            std = self.std
            mean = self.mean
        elif data_type == "face":
            std = self.face_std
            mean = self.face_mean
        elif data_type == "audio":
            std = self.audio_std
            mean = self.audio_mean
        else:
            assert False, f"datatype not defined: {data_type}"

        if torch.is_tensor(data):
            return data * torch.tensor(
                std, device=data.device, requires_grad=False
            ) + torch.tensor(mean, device=data.device, requires_grad=False)
        else:
            return data * std + mean

    def _pick_sequences(self, data_dict: Dict[str, Iterable], idx: List[int]) -> None:

        # max_len = max(len(sublist) for sublist in data_dict["data"])
        #
        # # 使用 np.pad 填充子列表
        # padded_data = np.array(
        #     [np.pad(sublist, (0, max_len - len(sublist)), mode='constant') for sublist in data_dict["data"]])

        # 处理data数据
        # 1. 首先找出第二维度的最大值
        max_second_dim = max(item.shape[1] for item in data_dict["data"])

        # 2. 正确填充到最大维度
        padded_list = []
        for item in data_dict["data"]:
            # 创建填充数组
            padded = np.zeros((9019, max_second_dim))
            # 复制原始数据
            padded[:item.shape[0], :item.shape[1]] = item
            padded_list.append(padded)

        # 3. 转换为最终数组
        final_array = np.array(padded_list)

        # missing_max_len = max(len(sublist) for sublist in data_dict["missing"])

        # 2. 正确填充到最大维度
        missing_padded_list = []
        for item in data_dict["missing"]:
            # 创建填充数组
            padded = np.zeros((9019, max_second_dim))
            # 复制原始数据
            padded[:item.shape[0], :item.shape[1]] = item
            missing_padded_list.append(padded)

        # 3. 转换为最终数组
        missing_final_array = np.array(missing_padded_list)

        try:
            self.audio = process_audio_tensors(data_dict["audio"], idx)

        except Exception as e:
            print(f"处理出错: {str(e)}")

            # 处理情绪数据
        try:
            if self.emotions is not None:
                self.emotions = process_emotion_tensors(self.emotions, idx)
                print(f"处理后的情绪批次形状: {self.emotions.shape}")
            else:
                print("没有情绪数据")
        except Exception as e:
            print(f"情绪数据处理出错: {str(e)}")


        self.data = np.take(final_array, idx, axis=0)
        self.missing = np.take(missing_final_array, idx, axis=0)
        self.audio = np.take(self.audio, idx, axis=0)
        #emotion input from np
        if self.emotions is not None:
            self.emotions = np.take(self.emotions, idx, axis=0)
        self.lengths = np.take(data_dict["lengths"], idx, axis=0)
        self.total_len = sum([len(d) for d in self.data])

    def _load_std(self) -> None:
        stats = torch.load(os.path.join(self.data_root, "data_stats.pth"))
        print(
            f'[dataset.py] loading from... {os.path.join(self.data_root, "data_stats.pth")}'
        )
        self.mean = stats["pose_mean"].reshape(-1)
        self.std = stats["pose_std"].reshape(-1)
        self.face_mean = stats["code_mean"]
        self.face_std = stats["code_std"]
        self.audio_mean = stats["audio_mean"]
        self.audio_std = stats["audio_std_flat"]

#分割数据
    def _chunk_data(self) -> None:
        chunk_data = []
        chunk_missing = []
        chunk_lengths = []
        chunk_audio = []
        chunk_emotions = []
        # create sequences of set lengths
        for d_idx in range(len(self.data)):
            curr_data = self.data[d_idx]
            curr_missing = self.missing[d_idx]
            curr_audio = self.audio[d_idx]
            end_range = len(self.data[d_idx]) - self.max_seq_length
            for chunk_idx in range(0, end_range, self.max_seq_length):
                chunk_end = chunk_idx + self.max_seq_length
                curr_data_chunk = curr_data[chunk_idx:chunk_end, :]
                curr_missing_chunk = curr_missing[chunk_idx:chunk_end, :]
                curr_audio_chunk = curr_audio[
                    chunk_idx * self.audio_per_frame : chunk_end * self.audio_per_frame,
                    :,
                ]
                if curr_data_chunk.shape[0] < self.max_seq_length:
                    # do not add a short chunk to the list
                    continue
                chunk_lengths.append(curr_data_chunk.shape[0])
                chunk_data.append(curr_data_chunk)
                chunk_missing.append(curr_missing_chunk)
                chunk_audio.append(curr_audio_chunk)
        idx = np.random.permutation(len(chunk_data))
        print("==> shuffle", idx)
        self.data = np.take(chunk_data, idx, axis=0)
        self.missing = np.take(chunk_missing, idx, axis=0)
        self.lengths = np.take(chunk_lengths, idx, axis=0)
        self.audio = np.take(chunk_audio, idx, axis=0)
        self.total_len = len(self.data)
        # ... 在循环中添加情绪数据的处理 ...
        if self.emotions is not None:
            self.emotions = np.take(chunk_emotions, idx, axis=0)


    def _register_keyframe_step(self) -> None:
        if self.add_frame_cond == 1:
            self.step = 30
        if self.add_frame_cond is None:
            self.step = 1

    def _pad_sequence(self, sequence,actual_length: int, max_length):
        if isinstance(sequence, np.ndarray):
            return _pad_numpy(sequence, max_length)
        elif isinstance(sequence, torch.Tensor):
            return _pad_tensor(sequence, max_length)
        else:
            raise TypeError(f"Unsupported sequence type: {type(sequence)}")

    def _get_idx(self, item: int) -> int:
        cumulative_len = 0
        seq_idx = 0
        while item > cumulative_len:
            cumulative_len += len(self.data[seq_idx])
            seq_idx += 1
        item = seq_idx - 1
        return item

    def _get_random_subsection(
        self, data_dict: Dict[str, Iterable]
    ) -> Dict[str, np.ndarray]:
        isnonzero = False
        while not isnonzero:
            start = np.random.randint(0, data_dict["m_length"] - self.max_seq_length)
            if self.add_padding:
                length = (
                    np.random.randint(self.min_seq_length, self.max_seq_length)
                    if not self.split == "test"
                    else self.max_seq_length
                )
            else:
                length = self.max_seq_length
            curr_missing = data_dict["missing"][start : start + length]
            isnonzero = np.any(curr_missing)
        missing = curr_missing
        motion = data_dict["motion"][start : start + length, :]
        emotions = data_dict.get("emotions", None)
        keyframes = motion[:: self.step]
        audio = data_dict["audio"][
            start * self.audio_per_frame : (start + length) * self.audio_per_frame,
            :,
        ]
        data_dict["m_length"] = len(motion)
        data_dict["k_length"] = len(keyframes)
        data_dict["a_length"] = len(audio)

        if data_dict["m_length"] < self.max_seq_length:
            motion = self._pad_sequence(
                motion, data_dict["m_length"], self.max_seq_length
            )
            missing = self._pad_sequence(
                missing, data_dict["m_length"], self.max_seq_length
            )
            audio = self._pad_sequence(
                audio, data_dict["a_length"], self.max_audio_length
            )
            max_step_length = len(np.zeros(self.max_seq_length)[:: self.step])
            keyframes = self._pad_sequence(
                keyframes, data_dict["k_length"], max_step_length
            )
            if emotions is not None:
                emotions = emotions[start: start + length]
                data_dict["e_length"] = len(emotions)
                emotions = self._pad_sequence(emotions, data_dict["e_length"], self.max_seq_length)
        data_dict["motion"] = motion
        data_dict["keyframes"] = keyframes
        data_dict["audio"] = audio
        data_dict["missing"] = missing
        data_dict["emotions"] = emotions
        return data_dict

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, item: int) -> Dict[str, np.ndarray]:
        # figure out which sequence to randomly sample from
        if not self.split == "test":
            item = self._get_idx(item)
        motion = self.data[item]
        audio = self.audio[item]
        # 在getitem中对情绪进行导入处理
        emotions = self.emotions[item] if self.emotions is not None else None

        e_length = len(emotions) if emotions is not None else 0
        m_length = self.lengths[item]
        missing = self.missing[item]
        a_length = len(audio)
        # Z Normalization
        if self.data_format == "pose":
            motion = (motion - self.mean) / self.std
        elif self.data_format == "face":
            motion = (motion - self.face_mean) / self.face_std
        audio = (audio - self.audio_mean) / self.audio_std
        keyframes = motion[:: self.step]
        k_length = len(keyframes)
        data_dict = {
            "motion": motion,
            "m_length": m_length,
            "audio": audio,
            "a_length": a_length,
            "keyframes": keyframes,
            "k_length": k_length,
            "missing": missing,
            "emotions": emotions,  # 添加这行
            "e_length": e_length,  # 添加这行
        }
        if not self.split == "test" and not self.chunk:
            data_dict = self._get_random_subsection(data_dict)
        if self.data_format == "face":
            data_dict["motion"] *= data_dict["missing"]
        return data_dict



# 填充到最长长度（返回一个批次张量）

def process_audio_tensors(tensor_list, idx):
    # 选择张量
    selected_tensors = [tensor_list[i] for i in idx]

    # 获取长度信息
    lengths = [tensor.shape[0] for tensor in selected_tensors]
    max_len = max(lengths)

    # 填充到最长长度
    processed_tensors = []
    for tensor in selected_tensors:
        if tensor.shape[0] < max_len:
            # 创建填充
            padding = torch.zeros((max_len - tensor.shape[0], tensor.shape[1]),
                                  dtype=tensor.dtype,
                                  device=tensor.device)
            # 拼接原始数据和填充
            padded = torch.cat([tensor, padding], dim=0)
            processed_tensors.append(padded)
        else:
            processed_tensors.append(tensor)

    # 堆叠成一个批次
    return torch.stack(processed_tensors)


def process_emotion_tensors(tensor_list, idx):

    # 选择张量
    selected_tensors = [tensor_list[i] for i in idx]

    # 获取长度信息
    lengths = [tensor.shape[0] for tensor in selected_tensors]
    max_len = max(lengths)

    # 填充到最长长度
    processed_tensors = []
    for tensor in selected_tensors:
        if tensor.shape[0] < max_len:
            # 创建填充
            padding = torch.zeros((max_len - tensor.shape[0], tensor.shape[1]), dtype=tensor.dtype,
                                  device=tensor.device)
            # 拼接原始数据和填充
            padded = torch.cat([tensor, padding], dim=0)
            processed_tensors.append(padded)
        else:
            processed_tensors.append(tensor)
    result = torch.stack(processed_tensors)

    return result


def _pad_numpy(sequence, max_length):
    actual_length = len(sequence)
    if actual_length >= max_length:
        return sequence[:max_length]
    else:
        if sequence.ndim == 1:
            padded = np.zeros(max_length, dtype=sequence.dtype)
        else:
            padded = np.zeros((max_length,) + sequence.shape[1:], dtype=sequence.dtype)
        padded[:actual_length] = sequence
        return padded


def _pad_tensor(sequence, max_length):
    actual_length = sequence.size(0)
    if actual_length >= max_length:
        return sequence[:max_length]
    else:
        if sequence.dim() == 1:
            padded = torch.zeros(max_length, dtype=sequence.dtype, device=sequence.device)
        else:
            padded = torch.zeros((max_length,) + tuple(sequence.shape[1:]), dtype=sequence.dtype,
                                 device=sequence.device)
        padded[:actual_length] = sequence
        return padded