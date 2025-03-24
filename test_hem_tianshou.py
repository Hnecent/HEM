from hem.HEM_v0 import aec_env
from hem.env.utils import Renderer

import yaml
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy

config_path = 'hem/env/config_env.yaml'
env_config = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))

env_attr = {
    'random_episode_split': True,
    'rolling_episode_split': False,
    'data_start_end': (24 * 60 * 2, 6 * 24 * 60),  # 数据集中，一共6，7，8，9月，122天，1分钟一个数据点
    'episode_time_steps': int(24 * 60 / env_config['MINUTES_PER_TIME_STEP']) * 2,  # one day, 24 hours
    'mode': 'train',
    'random_seed': 1,
    'noise_strength': 0,
    'config_path': config_path,
}

env = aec_env(**env_attr)

env = PettingZooEnv(env)

policy = RandomPolicy(action_space=env.action_space)

policies = MultiAgentPolicyManager(policies = [policy, policy, policy], env = env)

# Step 4: Convert the env to vector format
env = DummyVectorEnv([lambda: env])

# Step 5: Construct the Collector, which interfaces the policies with the vectorised environment
collector = Collector(policies, env)
collector.reset()

# Step 6: Execute the environment with the agents playing for 1 episode, and render a frame every 0.1 seconds
result = collector.collect(n_episode=2)
