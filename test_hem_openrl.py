from hem.HEM_v0 import env
import yaml
from openrl.envs.PettingZoo.registration import register
from openrl.envs.common import make
from openrl.selfplay.wrappers.random_opponent_wrapper import RandomOpponentWrapper

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

def HEMEnv(render_mode, **kwargs):
    return env(**env_attr)

register('HEM', HEMEnv)

# 创建单个环境的函数
def create_env():
    env = make("HEM")
    return env

# 创建环境函数列表
env_fns = [create_env for _ in range(10)]

# 检查各个环境的观察空间
for i, fn in enumerate(env_fns):
    print(i, fn)
    env = fn()
    print(f"环境 {i} 的观察空间: {env.observation_space}")
#
# # 创建向量化环境
# env = SyncVectorEnv(env_fns)
#
# env = make("HEM", env_num=10, opponent_wrappers=[RandomOpponentWrapper], )
#
# # 得到的该env，便可以直接用于OpenRL框架的训练了！
# from openrl.modules.common.ppo_net import PPONet as Net
# from openrl.runners.common.ppo_agent import PPOAgent as Agent
#
# agent = Agent(Net(env))  # 直接传入该环境即可
# agent.train(5000)  # 开始训练！
