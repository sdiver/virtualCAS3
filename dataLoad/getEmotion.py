import requests
import json
import os
import torch
import numpy as np
import time

def create_multipart_request(url):
    # Prepare the files
    files = []
    base_dir = os.path.dirname(url)  # Get directory path
    audio_filename = os.path.basename(url)  # Get filename

    filepath = os.path.join(base_dir, audio_filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")

    # Check if JSON file already exists
    json_filename = f"{os.path.splitext(audio_filename)[0]}_hume_api.json"
    json_filepath = os.path.join(base_dir, json_filename)

    if os.path.exists(json_filepath):
        try:
            with open(json_filepath, 'r') as f:
                data = json.load(f)
            if data:  # Check if the loaded data is not empty
                return json_filepath
            else:
                print("Loaded JSON file is empty. Proceeding to re-fetch data.")
        except json.JSONDecodeError:
            print("Error decoding existing JSON file. Proceeding to re-fetch data.")
    files.append(('file', (audio_filename, open(filepath, 'rb'))))
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

    job_id = parse_job_response(response)
    if job_id:
        max_retries = 30
        retry_interval = 10  # seconds

        for attempt in range(max_retries):
            result = job_preview(job_id)

            if result is None:
                print(
                    f"Failed to get job preview. Retrying in {retry_interval} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_interval)
                continue

            if isinstance(result, dict) and result.get('status') == 400 and result.get(
                    'message') == "Job is in progress.":
                print(
                    f"Job is still in progress. Retrying in {retry_interval} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_interval)
                continue

            if 'results' in result and 'predictions' in result['results']:
                # Save the result to a JSON file
                with open(json_filepath, 'w') as f:
                    json.dump(result, f)
                print(f"Saved API response to {json_filename}")
                return result

            print(
                f"Unexpected response format. Retrying in {retry_interval} seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_interval)

        print("Max retries reached. Job did not complete in time.")
    else:
        print("Failed to get job ID.")

    return None


def parse_job_response(response):
   try:
       return response.json().get('job_id')
   except ValueError:
       print("Error parsing JSON")
       return None

def job_preview(taskid):
    # Get job predictions (GET /v0/batch/jobs/:id/predictions)
    if taskid.startswith("./"):
        # Load and return JSON file
        try:
            with open(taskid, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            return {"status": 404, "message": f"File not found: {taskid}"}
        except json.JSONDecodeError:
            return {"status": 400, "message": f"Invalid JSON in file: {taskid}"}

    else:
        response = requests.get(
        f"https://api.hume.ai/v0/batch/jobs/{taskid}/predictions",  # Use f-string for proper string formatting
            headers={
                "X-Hume-Api-Key": "1sJm18G85hBifTpuITNw8OvubszP0rAdstFuf6EWfAP8I2a6"
            },
        )
        if response.status_code == 400:
            # Job is still in progress
            return {"status": 400, "message": "Job is in progress."}

        try:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0]
            else:
                print("Unexpected response format")
                print(f"Response content: {data}")
                return None
        except requests.exceptions.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Response content: {response.text}")
            return None


def process_audio_data(emotion_data, flip_person=False, audio_per_frame=1):
    """
    处理音频数据并对齐帧数

    Parameters:
        emotion_data: 情感数据
        flip_person: 是否需要翻转左右声道
        audio_per_frame: 每帧对应的音频采样点数

    Returns:
        audio_tensor: 处理后的音频张量
    """
    # 转换为情绪张量
    emotion_scores, _, _ = create_emotion_tensor(emotion_data)
    if emotion_scores is None:
        raise ValueError("Failed to create emotion tensor")

    curr_audio = torch.from_numpy(emotion_scores).float()

    # 如果需要翻转左右声道
    if flip_person:
        print("[get_data.py] flipping the dataset of left right person")
        tmp = torch.zeros_like(curr_audio)
        tmp[:, 1] = curr_audio[:, 0]  # 交换左右声道
        tmp[:, 0] = curr_audio[:, 1]
        curr_audio = tmp

    return curr_audio


def create_emotion_tensor(emotion_data):
    try:
        predictions = emotion_data.get('predictions', [])
        if not predictions:
            raise ValueError("No predictions found in emotion data")

        emotions = predictions[0].get('emotions', [])
        if not emotions:
            raise ValueError("No emotions found in predictions")

        # 创建一个字典来存储情绪分数
        emotion_dict = {emotion['name']: emotion['score'] for emotion in emotions}

        # 转换为numpy数组
        emotion_scores = np.array([emotion['score'] for emotion in emotions])

        # 获取情绪标签
        emotion_labels = [emotion['name'] for emotion in emotions]

        return emotion_scores, emotion_labels, emotion_dict
    except Exception as e:
        print(f"Error processing emotion data: {e}")
        return None, None, None

