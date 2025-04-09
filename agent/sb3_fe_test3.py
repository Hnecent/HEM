import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces


class TemporalAttention(nn.Module):
    """轻量级时间注意力机制"""

    def __init__(self, feature_dim, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        # 共享参数的多头注意力
        self.qkv = nn.Linear(feature_dim, 3 * feature_dim)
        self.proj = nn.Linear(feature_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        """
        输入形状: [batch_size, seq_len, feature_dim]
        输出形状: [batch_size, seq_len, feature_dim]
        """
        residual = x
        B, T, C = x.shape

        # 生成QKV
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, num_heads, T, head_dim]

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)

        # 残差连接
        x = self.proj(x)
        return self.layer_norm(x + residual)


class FastFeatureExtractorWithAttention(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, stack_size=5, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.stack_size = stack_size
        input_dim_per_frame = observation_space.shape[0] // stack_size

        # 1. 帧编码器（加入通道注意力）
        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(input_dim_per_frame),
            nn.Linear(input_dim_per_frame, 64),
            nn.ReLU(),
            # 轻量级通道注意力
            nn.Sequential(
                nn.Linear(64, 4),  # 压缩到4维
                nn.ReLU(),
                nn.Linear(4, 64),
                nn.Sigmoid()
            ),
            nn.LayerNorm(64)
        )

        # 2. 时间注意力层
        self.temporal_attn = TemporalAttention(feature_dim=64, num_heads=2)

        # 3. 特征聚合器（结合注意力上下文）
        self.temporal_mixer = nn.Sequential(
            nn.Linear(64 * 2, 128),  # 同时保留原始特征和注意力特征
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, features_dim),
            nn.LayerNorm(features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        # 重塑输入 [batch, stack_size * features] -> [batch, stack_size, features]
        x = observations.view(batch_size, self.stack_size, -1)

        # 逐帧编码 [batch, stack_size, 64]
        frame_features = self.frame_encoder(x)

        # 时间注意力 [batch, stack_size, 64]
        attn_features = self.temporal_attn(frame_features)

        # 特征聚合策略
        avg_pool = attn_features.mean(dim=1)  # 平均池化 [batch, 64]
        max_pool = attn_features.max(dim=1)[0]  # 最大池化 [batch, 64]
        concat_features = torch.cat([avg_pool, max_pool], dim=1)  # [batch, 128]

        # 最终映射
        return self.temporal_mixer(concat_features)
