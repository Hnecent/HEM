import numpy as np
import torch
import torch.nn.functional as F
from agent import rl_utils


def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)  # 施加正交初始化，提高训练稳定性
    torch.nn.init.constant_(layer.bias, bias_const)  # 将偏置初始化为常数
    return layer


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = _layer_init(torch.nn.Linear(state_dim, hidden_dim))
        self.fc2 = _layer_init(torch.nn.Linear(hidden_dim, action_dim))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, action_dim, net_arch):
        super(PolicyNetContinuous, self).__init__()

        # 构建深度网络
        layers = []
        input_dim = state_dim
        for hidden_dim in net_arch:
            layers.append(_layer_init(torch.nn.Linear(input_dim, hidden_dim)))
            layers.append(torch.nn.ReLU())
            input_dim = hidden_dim

        # 最终输出层
        self.hidden_layers = torch.nn.Sequential(*layers)
        self.fc_mu = _layer_init(torch.nn.Linear(input_dim, action_dim))
        self.fc_std = _layer_init(torch.nn.Linear(input_dim, action_dim))

    def forward(self, x):
        # 前向传播
        x = self.hidden_layers(x)
        mu = 1.0 * torch.tanh(self.fc_mu(x))  # 保持原有的动作范围限制 [-2, 2]
        std = F.softplus(self.fc_std(x)) + 1e-6

        # NaN 检测逻辑（保留原有调试机制）
        if torch.isnan(mu).any() or torch.isnan(std).any():
            print("NaN detected in mu or std!")
            print("mu:", mu)
            print("std:", std)

        return mu, std


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, net_arch):
        super(ValueNet, self).__init__()

        # 动态构建隐藏层
        layers = []
        input_dim = state_dim
        for hidden_dim in net_arch:
            layers.append(_layer_init(torch.nn.Linear(input_dim, hidden_dim)))
            layers.append(torch.nn.ReLU())
            input_dim = hidden_dim

        # 最终输出层（状态价值是标量）
        self.hidden_layers = torch.nn.Sequential(*layers)
        self.output_layer = _layer_init(torch.nn.Linear(input_dim, 1))

    def forward(self, x):
        x = self.hidden_layers(x)
        return self.output_layer(x)


class PPO:
    ''' PPO算法,采用截断方式 '''

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


class PPOContinuous:
    ''' 处理连续动作的PPO算法 '''

    def __init__(self, state_dim, action_dim, net_arch, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNetContinuous(state_dim, action_dim, net_arch).to(device)
        self.critic = ValueNet(state_dim, net_arch).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)

        # print(state, state.shape)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return action.detach().cpu().numpy()

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']),
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)

        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        if torch.isnan(states).any():
            print("NaN detected in states!")

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)

        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)

            log_probs = action_dists.log_prob(actions)

            ratio = torch.exp(log_probs - old_log_probs)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))

            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
