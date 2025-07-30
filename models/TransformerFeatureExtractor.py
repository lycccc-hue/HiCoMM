import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed

class TransformerFeatureExtractor(nn.Module):
    def __init__(self, dim=80):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans=3, embed_dim=dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=4),
            nn.TransformerEncoderLayer(d_model=dim, nhead=4)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.patch_embed(x)  # PatchEmbed 提取特征
        for layer in self.layers:
            x = layer(x)  # 仅使用 Transformer 处理
        return x

if __name__ == "__main__":
    model = TransformerFeatureExtractor()
    dummy_input = torch.randn(1, 3, 224, 224)
    features = model(dummy_input)
    print(features.shape)
