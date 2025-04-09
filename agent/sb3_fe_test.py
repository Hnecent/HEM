import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces


class InceptionResidualBlock(nn.Module):
    """带残差连接的多尺度Inception块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels // 4),
            nn.GELU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels // 4),
            nn.GELU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(out_channels // 4),
            nn.GELU()
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels // 4),
            nn.GELU()
        )

        # 残差连接适配层
        self.res_conv = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=(1, 1)) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.res_conv(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return out + residual  # 残差连接


class EnhancedFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces, stack_size=5, features_dim: int = 256, num_blocks: int = 3):
        super().__init__(observation_space, features_dim)
        self.stack_size = stack_size

        # 初始卷积层
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1)),  # 保持空间维度
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )

        # 多尺度特征提取层
        self.blocks = nn.Sequential()
        for i in range(num_blocks):
            self.blocks.add_module(
                f"block_{i}",
                nn.Sequential(
                    InceptionResidualBlock(64 if i == 0 else 128, 128),
                    nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),  # 宽度方向下采样
                    nn.BatchNorm2d(128),
                    nn.GELU()
                )
            )

        # 自适应池化+全连接
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, features_dim),
            nn.LayerNorm(features_dim),
            nn.Dropout(0.2)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入形状调整 [B, stack_size*D] -> [B, 1, stack_size, D]
        x = x.view(x.size(0), 1, self.stack_size, -1)

        # 特征提取
        x = self.init_conv(x)  # [B,64,5,36]
        x = self.blocks(x)  # [B,128,5,9]
        return self.final(x)  # [B,256]
