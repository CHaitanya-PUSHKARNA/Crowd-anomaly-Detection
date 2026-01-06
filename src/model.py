import torch.nn as nn

class CrowdCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, T, C, H, W)
        x = x[:, 0]  # fast sanity check using first frame
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


import torch
import torch.nn as nn

class CrowdCNNLSTM(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__()

        # CNN for spatial features
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.feature_dim = 64

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.size()

        # Merge batch & time
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = x.view(B, T, self.feature_dim)

        # LSTM
        _, (hn, _) = self.lstm(x)

        # Last hidden state
        out = self.fc(hn[-1])
        return out
