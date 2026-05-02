"""1D CNN 백본 + SL 분류기 + SSL 투영."""

import torch
import torch.nn as nn

from src.modulations import NUM_MODULATION_CLASSES


class CNNEncoder(nn.Module):
    def __init__(self, in_ch: int = 2, feat_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=3, padding=1),  # I/Q → 32ch
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # 추가 레이어
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # 길이 축 GAP
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return h.squeeze(-1)  # (B,C,1)→(B,C)


class SupervisedCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_MODULATION_CLASSES, feat_dim: int = 128):
        super().__init__()
        self.encoder = CNNEncoder(in_ch=2, feat_dim=feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)  # 로짓

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.classifier(z)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SSLModel(nn.Module):
    def __init__(self, feat_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        self.encoder = CNNEncoder(in_ch=2, feat_dim=feat_dim)
        self.projector = ProjectionHead(in_dim=feat_dim, proj_dim=proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        z = self.projector(h)
        return nn.functional.normalize(z, dim=1)  # contrastive용 단위벡터
