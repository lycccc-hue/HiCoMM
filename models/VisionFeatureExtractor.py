import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed

class FeatureExtractor(nn.Module):
    def __init__(self, dim=80, in_dim=32, depths=[1, 3, 8, 4], window_size=[8, 8, 14, 7]):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans=3, embed_dim=dim)
        self.layers = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1) if i < 2 else
            nn.TransformerEncoderLayer(d_model=dim, nhead=4) for i in range(len(depths))
        ])
        self.norm = nn.LayerNorm(dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.patch_embed(x)  # 输入图像通过PatchEmbed提取块
        # print(f"PatchEmbed output shape: {x.shape}")
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Conv2d):
                if x.dim() == 3:  # 如果 x 是 [B, N, C]
                    B, N, C = x.shape
                    H = W = int(N ** 0.5)  # 假设是正方形
                    x = x.reshape(B, C, H, W)  # 使用 reshape 变成 CNN 格式
                    # print(f"Reshaped x: {x.shape}")  # 确保是 [B, C, H, W]
                x = layer(x)  # 卷积层处理4D张量
                x = x.flatten(2).transpose(1, 2)  # 展平并转换为3D张量 [batch_size, seq_len, dim]
                # print(f"Reshaped for Transformer: {x.shape}")
            else:
                x = layer(x)  # Transformer编码器处理3D张量
        # print(f"Shape before avgpool: {x.shape}")  # 打印池化前的形状
        # x = x.permute(0, 2, 1).unsqueeze(-1)  # 变为 [batch_size, dim, seq_len, 1]
        # x = self.avgpool(x)  # 自适应池化得到单一特征
        # print(f"Shape after avgpool: {x.shape}")  # 检查池化后的形状
        #
        # x = x.flatten(1)  # 将池化后的特征展平为 [batch_size, dim]
        # print(f"Shape after view: {x.shape}")  # 检查展平后的形状
        #
        # # 确保x的形状为 [batch_size, dim]，传给 LayerNorm
        # if x.dim() == 2:  # 确保输入是 [batch_size, dim] 形状
        #     x = self.norm(x)  # 进行LayerNorm归一化
        # else:
        #     print(f"Unexpected shape: {x.shape}")
        return x


# 示例用法
if __name__ == "__main__":
    model = FeatureExtractor()
    dummy_input = torch.randn(1, 3, 224, 224)
    features = model(dummy_input)
    print(features.shape)  # 输出图像特征
