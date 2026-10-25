"""Attention over CNN+LSTM and ViT video embeddings (two branch tokens)."""
import torch
from torch import nn
from model.cnn_lstm import CNNLSTM
from model.vit import VideoViT

class AttentionFusion(nn.Module):
    def __init__(self, hidden_size=256, fusion_dim=256, heads=4, pretrained=True,
                 freeze_branches=False, mlp_hidden=128, mlp_dropout=0.3):
        super().__init__()
        if fusion_dim < 1 or heads < 1 or fusion_dim % heads:
            raise ValueError('fusion_dim must be positive and divisible by heads')
        if mlp_hidden < 1 or not 0 <= mlp_dropout < 1:
            raise ValueError('mlp_hidden must be positive; dropout must be in [0,1)')
        self.cnn_lstm = CNNLSTM(hidden_size, pretrained=pretrained)
        self.vit = VideoViT(pretrained=pretrained)
        # Branch classifiers are unused; preserve their keys for old checkpoints.
        self.cnn_lstm.classifier.requires_grad_(False)
        self.vit.classifier.requires_grad_(False)
        self.freeze_branches = freeze_branches
        if freeze_branches:
            self.cnn_lstm.requires_grad_(False)
            self.vit.requires_grad_(False)
        self.cnn_projection = nn.Sequential(nn.Linear(hidden_size, fusion_dim), nn.LayerNorm(fusion_dim))
        self.vit_projection = nn.Sequential(nn.Linear(768, fusion_dim), nn.LayerNorm(fusion_dim))
        self.branch_embedding = nn.Parameter(torch.empty(1, 2, fusion_dim))
        nn.init.normal_(self.branch_embedding, std=0.02)
        self.attention = nn.MultiheadAttention(fusion_dim, heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(fusion_dim)
        self.score = nn.Sequential(nn.Linear(fusion_dim, fusion_dim), nn.Tanh(), nn.Linear(fusion_dim, 1))
        # Raw logits for CrossEntropyLoss; softmax is applied only at inference.
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, 5),
        )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_branches:
            self.cnn_lstm.eval()
            self.vit.eval()
        return self

    def fuse_features(self, temporal, spatial, return_attention=False):
        tokens = torch.stack([self.cnn_projection(temporal), self.vit_projection(spatial)], dim=1)
        tokens = tokens + self.branch_embedding
        mixed, attention = self.attention(tokens, tokens, tokens, need_weights=return_attention,
                                          average_attn_weights=False)
        mixed = self.norm(tokens + mixed)
        weights = self.score(mixed).softmax(dim=1)
        logits = self.classifier((weights * mixed).sum(dim=1))
        if return_attention:
            return logits, {'branch_weights': weights.squeeze(-1), 'self_attention': attention}
        return logits

    def forward(self, videos, return_attention=False):
        b, t, c, h, w = videos.shape
        if t < 1 or (c, h, w) != (3, 224, 224):
            raise ValueError('Expected [B,T,3,224,224] with T>=1')
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_branches):
            frames = videos.reshape(b*t, c, h, w)
            cnn_features = self.cnn_lstm.backbone(frames)
            with torch.autocast(device_type=videos.device.type, enabled=False):
                _, (hidden, _) = self.cnn_lstm.lstm(cnn_features.float().reshape(b, t, -1))
            spatial = self.vit.backbone(frames).float().reshape(b, t, 768).mean(1)
        with torch.autocast(device_type=videos.device.type, enabled=False):
            return self.fuse_features(hidden[-1].float(), spatial.float(), return_attention)
