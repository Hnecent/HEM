"""Basic code which shows what it's like to run PPO on the Pistonball env using the parallel API, this code is inspired by CleanRL.

This code is exceedingly basic, with no logging or weights saving.
The intention was for users to have a (relatively clean) ~200 line file to refer to when they want to design their own learning algorithm.

Author: Jet (https://github.com/jjshoots)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import yaml

import hem.HEM_v0 as hem_v0
from hem.env.utils import Renderer

from supersuit import frame_stack_v1



class Agent(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            # nn.Flatten(),
            self._layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
        )

        self.critic = nn.Sequential(
            self._layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            self._layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, np.prod(action_shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_shape)))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)  # 施加正交初始化，提高训练稳定性
        torch.nn.init.constant_(layer.bias, bias_const)  # 将偏置初始化为常数
        return layer

    def get_value(self, x):
        return self.critic(self.feature_extractor(x))

    def get_action_and_value(self, x, action=None):
        x = self.feature_extractor(x)

        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    obs = np.stack([obs[a] for a in obs], axis=0)
    # transpose to be (batch, num_agents, *obs_shape)
    # obs = obs.reshape([-1, *obs.shape])
    # convert to torch
    obs = torch.tensor(obs).to(device)

    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)
    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}

    return x


if __name__ == "__main__":
    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 32
    stack_size = 2

    frame_size = (34 * stack_size,)
    max_cycles = 10000
    total_episodes = 100

    """ ENV SETUP """

    config_path = 'hem/env/config_env.yaml'
    env_config = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))

    env_attr = {
        'random_episode_split': True,
        'rolling_episode_split': False,
        'data_start_end': (24 * 60 * 2, 6 * 24 * 60),  # 数据集中，一共6，7，8，9月，122天，1分钟一个数据点
        'episode_time_steps': int(24 * 60 / env_config['MINUTES_PER_TIME_STEP']),  # one day, 24 hours
        'mode': 'train',
        'random_seed': 1,
        'noise_strength': 0,
        'config_path': config_path,
    }

    env = hem_v0.parallel_env(**env_attr)
    env = frame_stack_v1(env, stack_size=stack_size)

    num_agents = len(env.possible_agents)
    print("Number of agents: {}".format(num_agents))
    action_shape = env.action_space(env.possible_agents[0]).shape
    print("Shape of actions: {}".format(action_shape))
    observation_shape = env.observation_space(env.possible_agents[0]).shape
    print("Size of observations: {}".format(observation_shape))

    """ LEARNER SETUP """
    agent = Agent(obs_shape=observation_shape, action_shape=action_shape).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=0.001, eps=1e-5)

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0
    rb_obs = torch.zeros((max_cycles, num_agents, *frame_size)).to(device)
    rb_actions = torch.zeros((max_cycles, num_agents, *action_shape)).to(device)
    rb_logprobs = torch.zeros((max_cycles, num_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, num_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, num_agents)).to(device)
    rb_values = torch.zeros((max_cycles, num_agents)).to(device)

    """ TRAINING LOGIC """
    # train for n number of episodes
    for episode in range(total_episodes):
        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            next_obs, info = env.reset(seed=None)
            # reset the episodic return
            total_episodic_return = 0

            # each episode has num_steps
            for step in range(0, max_cycles):
                # rollover the observation
                obs = batchify_obs(next_obs, device)

                # get action from the agent
                actions, logprobs, _, values = agent.get_action_and_value(obs)

                # execute the environment and log data
                next_obs, rewards, terms, truncs, infos = env.step(
                    unbatchify(actions, env)
                )

                # add to episode storage
                rb_obs[step] = obs
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()

                # print(f'rb_rewards: {rb_rewards[step]}')
                # compute episodic return
                total_episodic_return += rb_rewards[step].cpu().numpy()
                # print(f'total_episodic_return: {total_episodic_return}')

                # if we reach termination or truncation, end
                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    end_step = step
                    break

        # bootstrap value if not done
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            for t in reversed(range(end_step)):
                delta = (
                        rb_rewards[t]
                        + gamma * rb_values[t + 1] * rb_terms[t + 1]
                        - rb_values[t]
                )
                rb_advantages[t] = delta + gamma * gamma * rb_advantages[t + 1]
            rb_returns = rb_advantages + rb_values

        # convert our episodes to batch of individual transitions
        b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
        b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
        print(f'b_actions.shape: {b_actions.shape}')
        b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))
        clip_fracs = []
        for epoch in range(100):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            for start in range(0, len(b_obs), batch_size):
                # select the indices we want to train on
                end = start + batch_size
                batch_index = b_index[start:end]

                _, newlogprob, entropy, value = agent.get_action_and_value(
                    b_obs[batch_index], b_actions[batch_index]
                )
                logratio = newlogprob - b_logprobs[batch_index]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                # normalize advantaegs
                advantages = b_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -b_advantages[batch_index] * ratio
                pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.flatten()
                v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                v_clipped = b_values[batch_index] + torch.clamp(
                    value - b_values[batch_index],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        print(f"Training episode {episode}")
        print(f"Episodic Return: {np.mean(total_episodic_return)}")
        print(f"Episode Length: {end_step}")
        print("")
        print(f"Value Loss: {v_loss.item()}")
        print(f"Policy Loss: {pg_loss.item()}")
        print(f"Old Approx KL: {old_approx_kl.item()}")
        print(f"Approx KL: {approx_kl.item()}")
        print(f"Clip Fraction: {np.mean(clip_fracs)}")
        print(f"Explained Variance: {explained_var.item()}")
        print("\n-------------------------------------------\n")

    """ RENDER THE POLICY """
    env = hem_v0.parallel_env(**env_attr)
    env = frame_stack_v1(env, stack_size=stack_size)
    renderer = Renderer(env=env.unwrapped, render_mode='episode')
    env.reset()
    agent.eval()

    with torch.no_grad():
        # render 5 episodes out
        for episode in range(1):
            obs, infos = env.reset(seed=None)
            obs = batchify_obs(obs, device)
            terms = {agent: False for agent in env.agents}
            truncs = {agent: False for agent in env.agents}
            while not any(terms.values()) and not any(truncs.values()):
                actions, logprobs, _, values = agent.get_action_and_value(obs)
                obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
                obs = batchify_obs(obs, device)
        renderer.render()
