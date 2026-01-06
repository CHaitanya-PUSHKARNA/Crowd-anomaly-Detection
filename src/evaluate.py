import torch
from torch.utils.data import DataLoader
from src.dataset import CrowdDataset
from src.model import CrowdCNN
from src.config import DATA_DIR, BATCH_SIZE

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CrowdDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    model = CrowdCNN().to(device)
    model.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for videos, labels in loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Accuracy: {correct / total:.2f}")

if __name__ == "__main__":
    evaluate()
