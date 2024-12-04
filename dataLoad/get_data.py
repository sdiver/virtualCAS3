"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os

from typing import Dict, List
from colorama import init, Fore, Back, Style
import numpy as np
import json
import torch
import torchaudio
from dataLoad.data import Social
from dataLoad.tensors import social_collate
from torch.utils.data import DataLoader
from utils.misc import prGreen
from dataLoad.getEmotion import create_multipart_request, job_preview
from scipy import interpolate

def get_dataset_loader(
    args,
    data_dict: Dict[str, np.ndarray],
    split: str = "train",
    chunk: bool = False,
    add_padding: bool = True,
) -> DataLoader:
    dataset = Social(
        args=args,
        data_dict=data_dict,
        split=split,
        chunk=chunk,
        add_padding=add_padding,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not split == "test",
        num_workers=8,
        drop_last=True,
        collate_fn=social_collate,
        pin_memory=True,
    )
    return loader


def _load_pose_data(
    all_paths: List[str], audio_per_frame: int, flip_person: bool = False
) -> Dict[str, List]:
    data = []
    face = []
    audio = []
    lengths = []
    missing = []
    # New list for text data
    emotions = []
    for _, curr_path_name in enumerate(all_paths):
        if not curr_path_name.endswith("_body_pose.npy"):
            continue
        # load face information and deal with missing codes
        curr_code = np.load(
            curr_path_name.replace("_body_pose.npy", "_face_expression.npy")
        ).astype(float)
        # curr_code = np.array(curr_face["codes"], dtype=float)
        missing_list = np.load(
            curr_path_name.replace("_body_pose.npy", "_missing_face_frames.npy")
        )
        if len(missing_list) == len(curr_code):
            print("skipping", curr_path_name, curr_code.shape)
            continue
        curr_missing = np.ones_like(curr_code)
        curr_missing[missing_list] = 0.0

        # load pose information and deal with discontinuities
        curr_pose = np.load(curr_path_name)
        if "PXB184" in curr_path_name or "RLW104" in curr_path_name:  # Capture 1 or 2
            curr_pose[:, 3] = (curr_pose[:, 3] + np.pi) % (2 * np.pi)
            curr_pose[:, 3] = (curr_pose[:, 3] + np.pi) % (2 * np.pi)

        # load audio information
        curr_audio, _ = torchaudio.load(
            curr_path_name.replace("_body_pose.npy", "_audio.wav")
        )

        curr_audio = curr_audio.T
        if flip_person:
            prGreen("[get_data.py] flipping the dataset of left right person")
            tmp = torch.zeros_like(curr_audio)
            tmp[:, 1] = curr_audio[:, 0]
            tmp[:, 0] = curr_audio[:, 1]
            curr_audio = tmp

        assert len(curr_pose) * audio_per_frame == len(
            curr_audio
        ), f"motion {curr_pose.shape} vs audio {curr_audio.shape}"

        '''
        增加情绪识别
        '''
        # 情感分析部分
        audio_path = curr_path_name.replace("_body_pose.npy", "_audio.wav")
        job_id = create_multipart_request(audio_path)
        job_preview_result = job_preview(job_id)


        # 安全地提取情感数据
        emotion_data = []
        try:
            if isinstance(job_preview_result, list) and len(job_preview_result) > 0:
                result = job_preview_result[0]
            elif isinstance(job_preview_result, dict):
                result = job_preview_result
            else:
                raise ValueError("Unexpected job_preview_result structure")

            predictions = result.get('results', {}).get('predictions', [])
            if predictions:
                burst_data = predictions[0].get('models', {}).get('burst', {})
                grouped_predictions = burst_data.get('grouped_predictions', [])
                if grouped_predictions:
                    emotion_data = grouped_predictions[0].get('predictions', [])

            if not emotion_data:
                prosody = predictions[0].get('models', {}).get('prosody', {})
                prosody_grouped_predictions = prosody.get('grouped_predictions', [])
                if prosody_grouped_predictions:
                    emotion_data = prosody_grouped_predictions[0].get('predictions', [])
            if not emotion_data:
                print(Fore.RED + f"No emotion data found for {curr_path_name}" + Style.RESET_ALL)
                continue

                # 打印提取的情感数据的摘要
            print(Fore.GREEN + f"Emotion data for {curr_path_name}: {len(emotion_data)} frames" + Style.RESET_ALL)

        except Exception as e:
            print(Fore.RED + f"Error extracting emotion data for {curr_path_name}:" + Style.RESET_ALL)
            print(Fore.RED + f"Error details: {str(e)}" + Style.RESET_ALL)
            # 如果无法提取情感数据，我们将跳过这个文件
            continue

        # 创建一个情绪名称列表
        if emotion_data and 'emotions' in emotion_data[0]:
            emotion_names = [emotion['name'] for emotion in emotion_data[0]['emotions']]
        else:
            print(Fore.RED + f"No emotions found in the first prediction for {curr_path_name}" + Style.RESET_ALL)
            continue

        # 提取情绪信息并转换为 tensor
        emotion_tensors = []
        timestamps = []

        for frame in emotion_data:
            if 'time' in frame and 'emotions' in frame:
                begin, end = frame['time'].get('begin', 0), frame['time'].get('end', 0)
                timestamps.append((begin, end))
                frame_emotions = {emotion['name']: emotion['score'] for emotion in frame['emotions']}
                frame_tensor = torch.tensor([frame_emotions.get(name, 0.0) for name in emotion_names],
                                            dtype=torch.float)
                emotion_tensors.append(frame_tensor)
            else:
                print(Fore.YELLOW + f"Skipping a frame due to missing data in {curr_path_name}" + Style.RESET_ALL)


        if not emotion_tensors:
            print(f"No valid emotion data found for {curr_path_name}")
            continue

        emotion_tensor = torch.stack(emotion_tensors)

        print(Fore.BLUE + f"Emotion tensor shape before interpolation: {emotion_tensor.shape}" + Style.RESET_ALL)
        print(
            Fore.BLUE + f"Emotion tensor contains NaN before interpolation: {torch.isnan(emotion_tensor).any()}" + Style.RESET_ALL)

        # 确保 emotion_tensor 的长度与姿势数据一致
        target_length = len(curr_pose)

        if emotion_tensor.shape[0] != target_length:

            # 计算音频时间戳
            audio_duration = len(curr_audio) / audio_per_frame
            pose_timestamps = np.linspace(0, audio_duration, len(curr_pose))

            # 使用情绪数据的开始时间作为参考
            old_timestamps = np.array([t[0] for t in timestamps])

            interpolated_emotions = []
            for i in range(emotion_tensor.shape[1]):
                # 创建插值函数
                f = interpolate.interp1d(old_timestamps, emotion_tensor[:, i].numpy(), kind='linear',
                                         fill_value='extrapolate')

                # 使用姿势数据的时间戳进行插值
                interpolated_channel = f(pose_timestamps)
                interpolated_emotions.append(interpolated_channel)

            # 将插值后的数据转换回tensor
            emotion_tensor = torch.tensor(np.array(interpolated_emotions).T, dtype=torch.float)
            print(Fore.CYAN + f"Emotion tensor shape after interpolation: {emotion_tensor.shape}" + Style.RESET_ALL)
            print(
                Fore.CYAN + f"Emotion tensor contains NaN after interpolation: {torch.isnan(emotion_tensor).any()}" + Style.RESET_ALL)

        # 验证数据长度的一致性
        assert len(curr_pose) == emotion_tensor.shape[
            0], f"Pose length {len(curr_pose)} does not match emotion tensor length {emotion_tensor.shape[0]}"
        assert len(curr_pose) * audio_per_frame == len(
            curr_audio), f"motion {curr_pose.shape} vs audio {curr_audio.shape}"

        # 检查 NaN 值
        if torch.isnan(emotion_tensor).any():
            print(
                Back.RED + Fore.WHITE + f"Warning: NaN values found in emotion tensor for {curr_path_name}" + Style.RESET_ALL)
            # 可以选择跳过这个样本或者用 0 替换 NaN 值
            emotion_tensor = torch.nan_to_num(emotion_tensor, nan=0.0)
            print(Fore.YELLOW + "NaN values have been replaced with 0.0" + Style.RESET_ALL)
        '''
        emotion analyse ended 
        '''


        data.append(curr_pose)
        face.append(curr_code)
        missing.append(curr_missing)
        audio.append(curr_audio)
        lengths.append(len(curr_pose))
        emotions.append(emotion_tensor)

    data_dict = {
        "data": data,
        "face": face,
        "audio": audio,
        "lengths": lengths,
        "missing": missing,
        "emotions": emotions,
    }
    return data_dict


def load_local_data(
    data_root: str, audio_per_frame: int, flip_person: bool = False
) -> Dict[str, List]:
    if flip_person:
        if "PXB184" in data_root:
            data_root = data_root.replace("PXB184", "RLW104")
        elif "RLW104" in data_root:
            data_root = data_root.replace("RLW104", "PXB184")
        elif "TXB805" in data_root:
            data_root = data_root.replace("TXB805", "GQS883")
        elif "GQS883" in data_root:
            data_root = data_root.replace("GQS883", "TXB805")

    all_paths = [os.path.join(data_root, x) for x in os.listdir(data_root)]
    all_paths.sort()
    return _load_pose_data(
        all_paths,
        audio_per_frame,
        flip_person=flip_person,
    )
