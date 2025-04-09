import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces


class FastFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, stack_size=5, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.stack_size = stack_size
        input_dim_per_frame = observation_space.shape[0] // stack_size

        # 1. 先将时序帧分开处理
        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(input_dim_per_frame),
            nn.Linear(input_dim_per_frame, 64),
            nn.ReLU()
        )

        # 2. 时序信息聚合 (可以捕获跨帧的时间特征)
        self.temporal_mixer = nn.Sequential(
            nn.Linear(64 * stack_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.LayerNorm(features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        # 将输入重塑为 [batch_size, stack_size, features_per_frame]
        x = observations.view(batch_size, self.stack_size, -1)

        # 对每个帧独立编码
        frame_features = []
        for i in range(self.stack_size):
            frame_features.append(self.frame_encoder(x[:, i, :]))

        # 将所有帧特征拼接并进行时序混合
        x = torch.cat(frame_features, dim=1)
        x = self.temporal_mixer(x)

        return x