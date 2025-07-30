import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import models.VisionFeatureExtractor as feature_extractor
from models.CNNFeatureExtractor import CNNFeatureExtractor
from models.TransformerFeatureExtractor import TransformerFeatureExtractor


class MultiHeadCoAttention(nn.Module):
    def __init__(self, num_heads=8, k=512):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = 768 // num_heads

        # 多头投影矩阵
        self.w_b_heads = nn.ModuleList([
            nn.Linear(768, 768, bias=False) for _ in range(num_heads)
        ])
        self.w_v = nn.Linear(768, k, bias=False)
        self.w_q = nn.Linear(768, k, bias=False)
        self.w_hv = nn.Linear(k, 1, bias=False)
        self.w_hq = nn.Linear(k, 1, bias=False)
        self.output_proj = nn.Linear(768 * num_heads, 768)
        self.dropout = nn.Dropout(0.1)

    def forward(self, text_hidden_states, image_hidden_states, text_mask):
        # 如果 image_hidden_states 为4D，转换为3D
        if image_hidden_states.dim() == 4:  # [batch, c, h, w]
            image_hidden_states = image_hidden_states.flatten(2).permute(0, 2, 1)  # [batch, h*w, c]

        assert text_hidden_states.dim() == 3, f"Text features should be 3D, got {text_hidden_states.dim()}"
        assert image_hidden_states.dim() == 3, f"Image features should be 3D, got {image_hidden_states.dim()}"
        all_head_outputs = []

        for head in range(self.num_heads):
            # 分头计算亲和矩阵
            projected_text = self.w_b_heads[head](text_hidden_states)
            affinity = torch.einsum('bxe,bye->bxy', projected_text, image_hidden_states)

            # 计算双路注意力
            wv_v = self.w_v(image_hidden_states)
            wq_q = self.w_q(text_hidden_states)

            # 跨模态特征增强
            wqqc = torch.einsum('bxk,bxy->byk', wq_q, affinity)
            wvvc = torch.einsum('byk,bxy->bxk', wv_v, affinity)

            # 注意力计算
            h_v = torch.tanh(wv_v + wqqc)
            h_q = torch.tanh(wq_q + wvvc)

            attention_v = F.softmax(self.w_hv(h_v).squeeze(2), dim=-1)
            attention_q = F.softmax(self.w_hq(h_q).squeeze(2), dim=-1)

            # 上下文聚合
            context_v = torch.einsum('bx,bxd->bd', attention_v, image_hidden_states)
            context_q = torch.einsum('by,byd->bd', attention_q, text_hidden_states)

            head_output = context_v + context_q
            all_head_outputs.append(head_output)

        combined = torch.cat(all_head_outputs, dim=-1)
        output = self.output_proj(combined)
        return output

class DynamicFusion(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, text_feat, image_feat):
        combined = torch.cat([text_feat, image_feat], dim=-1)
        gate_weights = self.gate(combined)
        fused = gate_weights[:, 0:1] * text_feat + gate_weights[:, 1:2] * image_feat
        return fused

class EnhancedVitBERT(nn.Module):
    def __init__(self, args,feature_type="full"):
        super().__init__()
        # 文本编码器
        self.bert = BertModel.from_pretrained(args.bert_pretrained_dir, return_dict=False)

        # 图像编码器
        if feature_type == "cnn":
            self.feature_extractor = CNNFeatureExtractor(dim=768)
        elif feature_type == "transformer":
            self.feature_extractor = TransformerFeatureExtractor(dim=768)
        else:
            self.feature_extractor = feature_extractor.FeatureExtractor(
                dim=768, in_dim=32, depths=[1, 3, 8, 4], window_size=[8, 8, 14, 7]
            )
        # 修改图像调整层：输入维度从256修改为768
        self.img_adjust = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(768, 768)
        )
        # self.fixed_image_vector = nn.Parameter(torch.randn(1, 768), requires_grad=True)

        # 多模态融合组件
        self.co_attention = MultiHeadCoAttention(num_heads=8)
        self.fusion = DynamicFusion()

        # 对比学习参数
        self.temperature = nn.Parameter(torch.tensor(0.07))

        # 分类器：简单线性层
        self.classifier = nn.Linear(768, 2)

    def contrastive_loss(self, text_feat, image_feat):
        text_norm = F.normalize(text_feat, p=2, dim=-1)
        image_norm = F.normalize(image_feat, p=2, dim=-1)
        logits = torch.matmul(text_norm, image_norm.T) / self.temperature
        labels = torch.arange(logits.size(0)).to(text_feat.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def forward(self, input_ids, vision_features, input_mask, added_attention_mask,
                hashtag_input_ids=None, hashtag_input_mask=None, labels=None):
        # 文本特征提取(Bert)
        text_output, _ = self.bert(input_ids=input_ids, attention_mask=input_mask)
        text_global = torch.mean(text_output, dim=1)  # [B,768]

        # 随机初始化模拟文本特征
        # batch_size = input_ids.shape[0]
        # text_output = torch.randn(batch_size, 50, 768).to(input_ids.device)  # 模拟 BERT 的输出
        # text_global = torch.mean(text_output, dim=1)  # [B, 768]

        # 图像特征提取：首先获取原始特征
        img_features_raw = self.feature_extractor(vision_features)  # 可能为 [B, N, 768]
        if img_features_raw.dim() == 3:
            B, N, C = img_features_raw.shape
            H = W = int(N ** 0.5)  # 假设patch数目构成正方形
            img_features_raw = img_features_raw.transpose(1, 2).reshape(B, C, H, W)  # [B,768,H,W]
        # 全局图像特征
        img_features_global = self.img_adjust(img_features_raw)  # [B,768]
        # 为跨模态注意力准备：扩展为 [B,1,768]
        img_features_for_attn = img_features_global.unsqueeze(1)
        # 用固定向量代替提取的图像特征
        # batch_size = text_global.shape[0]
        # img_features_global = self.fixed_image_vector.expand(batch_size, -1)  # [B, 768]

        # # 跨模态注意力计算
        co_attn_output = self.co_attention(text_output, img_features_for_attn, input_mask)

        # # 动态融合：使用全局图像特征（无额外维度）
        fused_features = self.fusion(text_global+ co_attn_output, img_features_global)
        # 直接使用 text_global + co_attn_output
        # fused_features = text_global + co_attn_output

        logits = self.classifier(fused_features)

        if labels is not None:
            cls_loss = F.cross_entropy(logits, labels)
            contrast_loss = self.contrastive_loss(text_global, img_features_global)
            total_loss = cls_loss + 0.3 * contrast_loss
            return total_loss
        return logits
