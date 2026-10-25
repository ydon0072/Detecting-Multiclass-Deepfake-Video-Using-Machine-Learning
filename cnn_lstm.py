import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class CNNLSTM(nn.Module):
    def __init__(self, hidden_size=256, pretrained=True, freeze_backbone=False):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
        self.lstm = nn.LSTM(features, hidden_size, batch_first=True)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden_size, 5))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()  # Preserve pretrained BatchNorm statistics.
        return self

    def forward(self, videos):
        b, t, c, h, w = videos.shape
        with torch.set_grad_enabled(self.training and not self.freeze_backbone):
            features = self.backbone(videos.reshape(b * t, c, h, w))
        # Float32 recurrent operations avoid backend-specific FP16 LSTM failures.
        with torch.autocast(device_type=videos.device.type, enabled=False):
            _, (hidden, _) = self.lstm(features.float().reshape(b, t, -1))
            return self.classifier(hidden[-1])
