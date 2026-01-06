# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader

# from src.dataset import CrowdDataset
# from src.model import CrowdCNN
# from src.config import DATA_DIR, BATCH_SIZE, EPOCHS, LR

# def train():
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     dataset = CrowdDataset(DATA_DIR)
#     loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#     model = CrowdCNN().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=LR)
#     criterion = nn.CrossEntropyLoss()

#     model.train()
#     for epoch in range(EPOCHS):
#         for videos, labels in loader:
#             videos, labels = videos.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(videos)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
#             break  # sanity run

# if __name__ == "__main__":
#     train()



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader

# from src.dataset import CrowdDataset
# from src.model import CrowdCNNLSTM
# from src.config import DATA_DIR, BATCH_SIZE, EPOCHS, LR

# def train():
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     dataset = CrowdDataset(DATA_DIR)
#     loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#     model = CrowdCNNLSTM().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=LR)
#     criterion = nn.CrossEntropyLoss()

#     model.train()
#     for epoch in range(EPOCHS):
#         for videos, labels in loader:
#             videos, labels = videos.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(videos)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
#             break  # sanity run



# if __name__ == "__main__":
#     train()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import CrowdDataset
from src.model import CrowdCNNLSTM
from src.config import DATA_DIR, BATCH_SIZE, EPOCHS, LR

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CrowdDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CrowdCNNLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        for videos, labels in loader:
            videos, labels = videos.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            break  # sanity run

    # ✅ ADD THIS BLOCK HERE (INSIDE train())
    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")


if __name__ == "__main__":
    train()
