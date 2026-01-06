import cv2
import torch
import numpy as np
from collections import deque

from src.model import CrowdCNNLSTM
from src.config import IMG_SIZE, MAX_FRAMES

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CrowdCNNLSTM()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

buffer = deque(maxlen=MAX_FRAMES)

cap = cv2.VideoCapture(0)  # 0 = webcam, or replace with video path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame_norm = frame_resized / 255.0
    buffer.append(frame_norm)

    if len(buffer) == MAX_FRAMES:
        clip = np.array(buffer)
        clip = torch.tensor(clip).permute(0, 3, 1, 2)
        clip = clip.unsqueeze(0).float().to(device)

        with torch.no_grad():
            out = model(clip)
            pred = torch.argmax(out, dim=1).item()

        label = "Violence" if pred == 0 else "Non-Violence"
        color = (0, 0, 255) if pred == 0 else (0, 255, 0)

        cv2.putText(
            frame,
            label,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

    cv2.imshow("Crowd Violence Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
