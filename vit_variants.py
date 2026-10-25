"""Frame-wise ViT-B/16 with mean feature pooling for video classification."""
import torch
from torch import nn
from torchvision.models import vit_b_16, vit_b_32, ViT_B_16_Weights, ViT_B_32_Weights

class VideoViT(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False, variant="vit_b_16"):
        super().__init__()
        builders = {"vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1), "vit_b_32": (vit_b_32, ViT_B_32_Weights.IMAGENET1K_V1)}
        builder, weights = builders[variant]
        self.backbone = builder(weights=weights if pretrained else None)
        self.backbone.heads = nn.Identity()
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(768, 5))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, videos):
        b, t, c, h, w = videos.shape
        if t < 1 or (c, h, w) != (3, 224, 224):
            raise ValueError('VideoViT expects [B, T, 3, 224, 224], T >= 1')
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_backbone):
            features = self.backbone(videos.reshape(b * t, c, h, w))
        # Mean pooling is permutation invariant; there is no temporal encoder.
        pooled = features.float().reshape(b, t, 768).mean(dim=1)
        return self.classifier(pooled)
