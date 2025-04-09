import torch
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, stack_size, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]
        self.features_extractor = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        output = self.features_extractor(observations)
        return output


class LookbackFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0] * observation_space.shape[1]
        self.flatten = nn.Flatten()
        self.features_extractor = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = self.flatten(observations)
        output = self.features_extractor(observations)
        return output


class RNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[1]
        self.rnn_input_dim = 32
        self.rnn_hidden_dim = 32
        self.rnn_num_layers = 2
        self.layer_norm = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.rnn_input_dim),
            nn.ReLU()
        )

        # self.rnn = nn.GRU(self.rnn_input_dim, self.rnn_hidden_dim, self.rnn_num_layers, batch_first=True)
        self.rnn = nn.LSTM(self.rnn_input_dim, self.rnn_hidden_dim, self.rnn_num_layers, batch_first=True)
        self.output_layer = nn.Sequential(
            nn.LayerNorm(self.rnn_hidden_dim),
            nn.Linear(self.rnn_hidden_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs = self.layer_norm(observations)
        obs, _ = self.rnn(obs)
        obs = obs[:, -1, :]
        output = self.output_layer(obs)
        return output


class CNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        l_dim = observation_space.shape[0]
        f_dim = observation_space.shape[1]

        self.cnn = nn.Sequential(
            nn.LayerNorm([f_dim, l_dim]),
            nn.Conv1d(in_channels=f_dim, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float().permute(0, 2, 1)).shape[1]
        self.output = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs = observations.permute(0, 2, 1)  # Change the shape to (Batch, features, length)
        return self.output(self.cnn(obs))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, d_model=512, n_head=8, num_encoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, features_dim=256):
        super(TransformerEncoderFeatureExtractor, self).__init__(observation_space, features_dim)

        input_dim = observation_space.shape[1]

        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * observation_space.shape[0], features_dim),
            nn.ReLU(),
        )

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.flatten(output)
        return output


class LinearAttentionLayer(nn.Module):
    """线性注意力层（特征级）"""

    def __init__(self, embed_dim, reduction_ratio=4):
        """
        Args:
            embed_dim: 输入特征维度
            reduction_ratio: 降维比例（平衡计算量与表达能力）
        """
        super().__init__()
        self.reduction_dim = embed_dim // reduction_ratio

        # 注意力权重生成
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, self.reduction_dim),
            nn.ReLU(),
            nn.Linear(self.reduction_dim, embed_dim),
            nn.Sigmoid()  # 输出0-1的注意力权重
        )

    def forward(self, x):
        """
        输入: [batch_size, embed_dim]
        输出: [batch_size, embed_dim]
        """
        # 生成注意力权重 [B, D]
        attn_weights = self.attention(x)

        # 特征重校准
        return x * attn_weights  # 逐元素乘法


class EnhancedFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        print(observation_space)
        input_dim = observation_space.shape[0]

        # 特征提取主干网络
        self.main_branch = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 线性注意力增强模块
        self.attention_branch = nn.Sequential(
            LinearAttentionLayer(features_dim),
            nn.Linear(features_dim, features_dim),  # 注意力后的特征变换
            nn.ReLU()
        )

        # 残差连接
        self.residual = nn.Linear(input_dim, features_dim) if input_dim != features_dim else nn.Identity()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        输入: [batch_size, input_dim]
        输出: [batch_size, features_dim]
        """
        # 主干网络
        main_features = self.main_branch(observations)  # [B, D]

        # 注意力增强
        attended_features = self.attention_branch(main_features)

        # 残差连接
        residual = self.residual(observations)

        return attended_features + residual  # [B, D]
