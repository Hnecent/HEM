# my package
from hem.HEM_v0 import parallel_env, sa_env, gym_env
from hem.HEM_v1 import sa_stack_env
from hem.env.objects.data import Data
from agent.sb3_fe import FeaturesExtractor, EnhancedFeaturesExtractor

# stable_baselines3
from stable_baselines3 import SAC, PPO, DDPG
from stable_baselines3.common.callbacks import EvalCallback

# other package
import torch
import os
import pickle
import time
import datetime
import logging
import yaml
import supersuit as ss

config_path = 'hem/env/config_env.yaml'
env_config = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))

# important parameters
# agent
sb3_model = SAC  # PPO, A2C, DDPG, TD3, SAC
net_arch = [256, 512, 128, 32, 128, 512, 256]  # the architecture of the policy and value networks
features_extractor_kwargs = dict(features_dim=256)  # the output dimension of the features extractor
features_extractor_class = FeaturesExtractor

# env
env_cl = sa_env
episode_time_steps = int(24 * 60 / env_config['MINUTES_PER_TIME_STEP'])  # 2 days
data_start_end = (0, 122 * 24 * 60)  # 2 days

# training
val_freq = episode_time_steps * 5
log_interval = 1


def checkpoint_path(experiment_name, iteration_count, sample_count):
    dir_path = f'./checkpoint/{experiment_name}/iteration_{iteration_count}/sample_{sample_count}'
    os.makedirs(dir_path, exist_ok=True)
    checkpoint_path_dict = {
        'tensorboard_log': dir_path,
        'best_model_save_path': dir_path,
        'log_path': dir_path,
        'env_set_path': dir_path + '/env_set.pkl',
        'render_csv_path': dir_path + '/render.csv',
    }
    return checkpoint_path_dict


def make_train_eval_env(checkpoint_path_dict, random_seed):
    train_env_kwargs = {
        'random_episode_split': True,
        'rolling_episode_split': False,
        'data_start_end': data_start_end,  # 数据集中，一共6，7，8，9月，122天，1分钟一个数据点
        'episode_time_steps': episode_time_steps,  # one day, 24 hours
        'mode': 'train',
        'random_seed': random_seed,
        'noise_strength': 0,
        'config_path': config_path,
        'render_mode': 'None',
    }

    eval_env_kwargs = train_env_kwargs.copy()
    eval_env_kwargs['mode'] = 'val'
    # eval_env_kwargs['mode'] = 'train'

    raw_train_env = env_cl(**train_env_kwargs)
    raw_eval_env = env_cl(**eval_env_kwargs)

    if raw_train_env.metadata['multi_agent']:
        train_env = ss.pettingzoo_env_to_vec_env_v1(raw_train_env)
        train_env = ss.concat_vec_envs_v1(train_env, 14, num_cpus=14, base_class="stable_baselines3")

        eval_env = ss.pettingzoo_env_to_vec_env_v1(raw_eval_env)
        eval_env = ss.concat_vec_envs_v1(eval_env, 1, num_cpus=14, base_class="stable_baselines3")
    else:
        train_env = ss.stable_baselines3_vec_env_v0(raw_train_env, num_envs=14)
        eval_env = ss.stable_baselines3_vec_env_v0(raw_eval_env, num_envs=1)

    # eval_env = Monitor(eval_env)
    with open(checkpoint_path_dict['env_set_path'], 'wb') as f:
        pickle.dump({'train_env_kwargs': train_env_kwargs, 'eval_env_kwargs': eval_env_kwargs}, f)
    return train_env, eval_env


def load_train_eval_data(minutes_per_time_step: int):

    data_kwargs = {
        'data_start_end': data_start_end,  # 数据集中，一共6，7，8，9月，122天，1分钟一个数据点
        'minutes_per_time_step': minutes_per_time_step,
        'random_seed': 42,
        'noise_strength': 0,
    }

    train_data = Data(mode='train',**data_kwargs)
    eval_data = Data(mode='val', **data_kwargs)

    return train_data, eval_data


class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, total_timesteps, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.verbose >= 1:
                percent_complete = (self.num_timesteps / self.total_timesteps) * 100
                print(f"Percent complete: {percent_complete:.2f}%")
                self.logger.record("eval/percent_complete", percent_complete)
        return result


def train_sb3_model(train_env, eval_env, train_timesteps, checkpoint_path_dict, device):
    eval_callback = CustomEvalCallback(
        total_timesteps=train_timesteps,
        eval_env=eval_env,
        n_eval_episodes=3,  # 由于sb3使用多个向量环境去模拟多智能体系统，如果是多智能体环境，必须大于num_agents, 而且最好是num_agents的整数倍
        eval_freq=val_freq,
        log_path=checkpoint_path_dict['log_path'],
        best_model_save_path=checkpoint_path_dict['best_model_save_path'],
        deterministic=True,
        render=False,
        verbose=1
    )

    train_kwargs = {
        'total_timesteps': train_timesteps,
        'callback': eval_callback,
        'log_interval': log_interval,
    }

    model = sb3_model(
        policy='MlpPolicy',
        env=train_env,
        policy_kwargs=dict(
            net_arch=net_arch,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
        ),
        tensorboard_log=checkpoint_path_dict['tensorboard_log'],
        verbose=0,
        device=device)

    model.learn(**train_kwargs)
    return model


class StreamToLogger(object):
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip() != "":  # 忽略空消息
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass


def train_pipline(user_experiment_name, train_timesteps, device):
    """
    :param user_experiment_name: str, the name of the experiment
    :return:
    """
    experiment_start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f'{user_experiment_name}_{experiment_start_time}'
    # 设置日志
    logging.basicConfig(
        filename=f'log/{experiment_name}.log',  # 日志文件名
        level=logging.INFO,  # 设置日志级别为INFO
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 设置日志重定向
    # sys.stdout = StreamToLogger(logging, logging.INFO)
    print(f"{'=' * 20} {experiment_name} {'=' * 20}")

    checkpoint_path_dict = checkpoint_path(experiment_name, 0, 0)
    train_env, eval_env = make_train_eval_env(checkpoint_path_dict, 42)

    # train
    start_time = time.time()
    last_agent = train_sb3_model(train_env, eval_env, train_timesteps, checkpoint_path_dict, device)
    print(f'Training time: {time.time() - start_time}')


if __name__ == '__main__':
    device_str = input("输入测试设备（cuda, cpu, mps）：")
    device = torch.device(device_str)
    train_timesteps = episode_time_steps * 10000

    """
    命名规则：env_model_device
    """

    user_experiment_name = 'sa_stack_sac_washer_fe'
    train_pipline(user_experiment_name, train_timesteps, device)
