from pettingzoo.utils import aec_to_parallel

from hem.HEM_v0 import aec_env, parallel_env, sa_env
import yaml
from hem.env.utils import Renderer
from pettingzoo.test import api_test, parallel_api_test

import supersuit as ss
from pettingzoo.utils.conversions import aec_to_parallel

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

# print(f"{'-' * 20} Testing aec_env {'-' * 20}")
# env = aec_env(**env_attr)
# api_test(env, num_cycles=50)
#
# renderer = Renderer(env=env.unwrapped, render_mode='episode')
#
# for e in range(10):
#     env.reset()
#     for agent in env.agent_iter(max_iter=1e5):
#         observation, reward, termination, truncation, infos = env.last()
#
#         # print(f'agent: {agent}, observation: {observation}, reward: {reward}, termination: {termination}, truncation: {truncation}, infos: {infos}')
#         # print(env._cumulative_rewards)
#
#         if termination or truncation:
#             action = None
#             break
#         else:
#             action = env.action_space(agent).sample()
#         env.step(action)
# renderer.render()
# env.close()
#
# print(f"{'-' * 20} Testing parallel_env {'-' * 20}")
# env = parallel_env(**env_attr)
# parallel_api_test(env, num_cycles=50)
#
# renderer = Renderer(env=env.unwrapped, render_mode='episode')
#
# for e in range(10):
#     observation, infos = env.reset()
#     terminations = {agent: False for agent in env.agents}
#     truncations = {agent: False for agent in env.agents}
#     while not any(terminations.values()) and not any(truncations.values()):
#         actions = {agent: env.action_space(agent).sample() for agent in env.agents}
#         observation, rewards, terminations, truncations, infos = env.step(actions)
#         # print(f'observation: {observation}, rewards: {rewards}, terminations: {terminations}, truncations: {truncations}, infos: {infos}')
# renderer.render()
# env.close()
#
#
# print(f"{'-' * 20} Testing sa_env {'-' * 20}")
# env = sa_env(**env_attr)
# parallel_api_test(env, num_cycles=50)
#
# renderer = Renderer(env=env.unwrapped, render_mode='episode')
#
# for e in range(4):
#     observation, infos = env.reset()
#     terminations = {agent: False for agent in env.agents}
#     truncations = {agent: False for agent in env.agents}
#     while not any(terminations.values()) and not any(truncations.values()):
#         actions = {agent: env.action_space(agent).sample() for agent in env.agents}
#         observation, rewards, terminations, truncations, infos = env.step(actions)
#         # print(f'observation: {observation}, rewards: {rewards}, terminations: {terminations}, truncations: {truncations}, infos: {infos}')
# renderer.render()
# env.close()


print(f"{'-' * 20} Testing Vec env {'-' * 20}")
env = sa_env(**env_attr)
env = ss.pettingzoo_env_to_vec_env_v1(env)

obs, _ = env.reset()

print(obs, obs.shape)
