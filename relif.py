import os
import numpy as np
import cv2
import torch
from scipy.io import wavfile
from model.diffusion import FiLMTransformer
from utils.misc import fix_seed
from utils.model_util import create_model_and_diffusion, load_model
from visualize.render_codes import BodyRenderer

class VideoGenerator:
    def __init__(self, config_base):
        self.body_renderer = BodyRenderer(
            config_base=config_base,
            render_rgb=True,
        )

    def generate_video_from_data(self, data_dir):
        """
        Generate video from data files in the given directory.

        Args:
        data_dir (str): Path to the directory containing data files.

        Returns:
        None
        """
        # Load face motion data
        face_motion = np.load(os.path.join(data_dir, 'scene01_face_expression.npy'))

        # Load pose data
        pose = np.load(os.path.join(data_dir, 'scene01_body_pose.npy'))

        # Load audio data
        sample_rate, audio = wavfile.read(os.path.join(data_dir, 'scene01_audio.wav'))

        # Prepare data for rendering
        render_data_block = {
            "audio": audio.astype(np.float32),
            "body_motion": pose,
            "face_motion": face_motion,
        }

        # Render the full video
        self.body_renderer.render_full_video(
            render_data_block,
            os.path.join(data_dir, "output"),
            audio_sr=sample_rate
        )

        print("Video generated successfully!")

# Example usage
config_base = "./checkpoints/ca_body/data/PXB184"
video_generator = VideoGenerator(config_base)
video_generator.generate_video_from_data('data/GQS883')
