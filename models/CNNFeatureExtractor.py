import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed

class CNNFeatureExtractor(nn.Module):
    def __init__(self, dim=80):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans=3, embed_dim=dim)
        self.layers = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        ])
        self.norm = nn.LayerNorm(dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.patch_embed(x)  # 图像转PatchEmbed
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.reshape(B, C, H, W)  # 变为CNN格式

        for layer in self.layers:
            x = layer(x)

        x = x.flatten(2).transpose(1, 2)  # 转回 Transformer 兼容格式
        return x

if __name__ == "__main__":
    model = CNNFeatureExtractor()
    dummy_input = torch.randn(1, 3, 224, 224)
    features = model(dummy_input)
    print(features.shape)
