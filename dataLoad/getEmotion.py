import requests
import json
import os
import torch
import numpy as np

def create_multipart_request(url):
    # Prepare the files
    files = []
    base_dir = os.path.dirname(url)  # Get directory path
    audio_files = os.path.basename(url)  # Get filename


    for filename in audio_files:
        filepath = os.path.join(base_dir, filename)
        files.append(('file', (filename, open(filepath, 'rb'))))

    # Prepare the request
    response = requests.post(
        "https://api.hume.ai/v0/batch/jobs",
        headers={
            "X-Hume-Api-Key": "1sJm18G85hBifTpuITNw8OvubszP0rAdstFuf6EWfAP8I2a6"
        },
        data={
            'json': json.dumps({
                "models": {
                    "burst": {},
                    "prosody": {},
                    "language": {},
                    "ner": {}
                }
            }),
        },
        files=files
    )

    return parse_job_response(response)

def parse_job_response(response):
   try:
       return response.json().get('job_id')
   except ValueError:
       print("Error parsing JSON")
       return None

def job_preview(taskid):
    # Get job predictions (GET /v0/batch/jobs/:id/predictions)
    response = requests.get(
        f"https://api.hume.ai/v0/batch/jobs/{taskid}/predictions",  # Use f-string for proper string formatting
        headers={
            "X-Hume-Api-Key": "1sJm18G85hBifTpuITNw8OvubszP0rAdstFuf6EWfAP8I2a6"
        },
    )
    return response.json()


def process_audio_data(job_result, flip_person=False, audio_per_frame=1):
    """
    处理音频数据并对齐帧数

    Parameters:
        audio_path: 音频文件路径
        flip_person: 是否需要翻转左右声道
        audio_per_frame: 每帧对应的音频采样点数

    Returns:
        audio_tensor: 处理后的音频张量
    """
    # 2. 转换为情绪张量
    emotion_scores, _, _ = create_emotion_tensor(job_result)
    curr_audio = torch.from_numpy(emotion_scores).float()

    # 3. 如果需要翻转左右声道
    if flip_person:
        print("[get_data.py] flipping the dataset of left right person")
        tmp = torch.zeros_like(curr_audio)
        tmp[:, 1] = curr_audio[:, 0]  # 交换左右声道
        tmp[:, 0] = curr_audio[:, 1]
        curr_audio = tmp

    # 4. 检查音频长度与帧数是否对齐
    def check_alignment(curr_audio, curr_pose_len, audio_per_frame):
        expected_audio_len = curr_pose_len * audio_per_frame
        assert len(curr_audio) == expected_audio_len, \
            f"motion frames {curr_pose_len} vs audio samples {len(curr_audio)}"

    return curr_audio


def create_emotion_tensor(response_json):
    # 获取情绪预测数组
    try:
        predictions = \
        response_json[0]['results']['predictions'][0]['models']['burst']['grouped_predictions'][0]['predictions'][0][
            'emotions']

        # 创建一个字典来存储情绪分数
        emotion_dict = {emotion['name']: emotion['score'] for emotion in predictions}

        # 转换为numpy数组
        emotion_scores = np.array([emotion['score'] for emotion in predictions])

        # 获取情绪标签
        emotion_labels = [emotion['name'] for emotion in predictions]

        return emotion_scores, emotion_labels, emotion_dict
    except Exception as e:
        print(f"Error processing response: {e}")
        return None, None, None

