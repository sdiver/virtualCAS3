"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os

from typing import Dict, List

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
        # ... existing code ...

        '''
        增加情绪识别
        '''
        # 获取任务jobid
        job_id = create_multipart_request(curr_path_name)

        # 查询jobid识别后的结果
        job_preview_result = job_preview(job_id)

        # 解析 JSON 数据
        response_json = json.loads(job_preview_result)

        emotion_data = response_json[0]['results']['predictions'][0]['models']['burst']['grouped_predictions'][0][
            'predictions']

        # 创建一个情绪名称列表
        emotion_names = [emotion['name'] for emotion in emotion_data[0]['emotions']]

        print(emotion_names)

        # 提取情绪信息并转换为 tensor
        emotion_tensors = []
        timestamps = []
        frame_durations = []

        # 遍历整个emotions的结果
        for frame in emotion_data:
            begin, end = frame['time']['begin'], frame['time']['end']
            duration = end - begin
            timestamps.append((begin, end))
            frame_emotions = {emotion['name']: emotion['score'] for emotion in frame['emotions']}
            frame_tensor = torch.tensor([frame_emotions.get(name, 0.0) for name in emotion_names], dtype=torch.float)
            emotion_tensors.append(frame_tensor)
            frame_durations.append(duration)

        # 将emotion_tensors列表转换为tensor
        emotion_tensor = torch.stack(emotion_tensors)

        # 确保 emotion_tensor 的长度与姿势数据一致
        target_length = len(curr_pose)

        if emotion_tensor.shape[0] != target_length:
            print(
                f"Adjusting emotion tensor length from {emotion_tensor.shape[0]} to match pose data length {target_length}")

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

        # 验证数据长度的一致性
        assert len(curr_pose) == emotion_tensor.shape[
            0], f"Pose length {len(curr_pose)} does not match emotion tensor length {emotion_tensor.shape[0]}"
        assert len(curr_pose) * audio_per_frame == len(
            curr_audio), f"motion {curr_pose.shape} vs audio {curr_audio.shape}"

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
