"""Uses Stable-Baselines3 to train agents to play the Waterworld environment using SuperSuit vector envs.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""
from __future__ import annotations

import glob
import os
import time

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import yaml
from hem import HEM_v0
from hem.env.utils import Renderer


def train_butterfly_supersuit(env, steps: int = 10_000, seed: int | None = 0):
    # Train a single model to play as each agent in a cooperative Parallel environment
    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")

    model = PPO(
        MlpPolicy,
        env,
        verbose=3,
        learning_rate=1e-3,
        batch_size=256,
    )

    model.learn(total_timesteps=steps)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def eval(env, num_games: int = 100, render: bool = True):
    # Evaluate a trained agent vs a random agent

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games})"
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent
    # for i in range(num_games):
    #     env.reset(seed=i)
    #
    #     for agent in env.agent_iter():
    #         obs, reward, termination, truncation, info = env.last()
    #
    #         for a in env.agents:
    #             rewards[a] += env.rewards[a]
    #         if termination or truncation:
    #             break
    #         else:
    #             act = model.predict(obs, deterministic=True)[0]
    #
    #         env.step(act)
    # env.close()
    renderer = Renderer(env=env.unwrapped, render_mode='episode')

    for e in range(num_games):
        observation, infos = env.reset()
        terminations = {agent: False for agent in env.agents}
        truncations = {agent: False for agent in env.agents}
        while not any(terminations.values()) and not any(truncations.values()):
            actions = model.predict(observation)

            observation, rewards, terminations, truncations, infos = env.step(actions)
            # print(f'observation: {observation}, rewards: {rewards}, terminations: {terminations}, truncations: {truncations}, infos: {infos}')
        if render:
            renderer.render()
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    return avg_reward


if __name__ == "__main__":
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
        'render_mode': 'None',
    }

    train_env = HEM_v0.sa_env(**env_attr)
    test_env = HEM_v0.sa_env(**env_attr)


    # Train a model (takes ~3 minutes on GPU)
    train_butterfly_supersuit(train_env, steps=196_608, seed=0)

    eval(test_env, num_games=10, render=True)
