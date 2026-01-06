import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from src.config import IMG_SIZE, MAX_FRAMES, CLASSES

class CrowdDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for label, cls in enumerate(CLASSES):
            cls_path = os.path.join(root_dir, cls)
            for video in os.listdir(cls_path):
                self.samples.append((os.path.join(cls_path, video), label))

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []

        while len(frames) < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame / 255.0
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"Empty video: {path}")

        while len(frames) < MAX_FRAMES:
            frames.append(frames[-1])

        frames = np.array(frames)
        frames = torch.tensor(frames).permute(0, 3, 1, 2)

        return frames.float()

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        video = self.load_video(video_path)
        return video, torch.tensor(label)

    def __len__(self):
        return len(self.samples)
