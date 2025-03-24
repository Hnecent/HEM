import numpy as np
import matplotlib.pyplot as plt
import pickle
import supersuit as ss


def plot_log(log_path, env_set_path):
    log = np.load(log_path, allow_pickle=True)
    train_env_kwargs = np.load(env_set_path, allow_pickle=True)['train_env_kwargs']
    x = log['timesteps'] / (train_env_kwargs['episode_time_steps'])
    y = np.mean(log['results'], axis=1)

    # 定义移动平均函数
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    # 平滑数据
    window_size = 5
    y_smooth = moving_average(y, window_size)

    # 绘制原始数据和平滑后的数据
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Original Data', alpha=0.5)
    plt.plot(x[(window_size - 1):], y_smooth, label='Smoothed Data', color='red')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Evaluation Rewards')
    plt.title('Evaluation Rewards Over Episodes')
    plt.grid(True)
    plt.legend()
    plt.show()


def load_best_model(checkpoint_path_dict, sb3_model):
    model = sb3_model.load(checkpoint_path_dict['best_model_save_path'] + '/best_model.zip')
    print(f'Load best model from {checkpoint_path_dict["best_model_save_path"]}')
    return model


def load_test_env(checkpoint_path_dict, env_cl, num_episodes_in_one_test, train_mode=False, noise_strength=0, ):
    with open(checkpoint_path_dict['env_set_path'], 'rb') as f:
        env_set = pickle.load(f)
    test_env_kwargs = env_set['eval_env_kwargs']
    if train_mode:
        test_env_kwargs['mode'] = 'train'
    else:
        test_env_kwargs['mode'] = 'test'

    test_env_kwargs['noise_strength'] = noise_strength
    test_env_kwargs['random_episode_split'] = False
    test_env_kwargs['episode_time_steps'] = env_set['eval_env_kwargs']['episode_time_steps'] * num_episodes_in_one_test

    test_env = env_cl(**test_env_kwargs)
    if test_env.metadata['multi_agent'] == True:
        test_env = ss.pettingzoo_env_to_vec_env_v1(test_env)
    else:
        test_env = ss.stable_baselines3_vec_env_v0(test_env, num_envs=1)
    return test_env


def pz_simulation_process(env, renderer, model, num_episodes):
    obs, infos = env.reset()
    for e in range(num_episodes):
        renderer.reset()
        terminations = [False]
        truncations = [False]
        while not any(terminations) and not any(truncations):
            # print('obs:', obs)
            act, _ = model.predict(obs)
            # print('m_act:', act)
            # act = obs * 2 / 3 - 1
            # print('act:', act)
            obs, rewards, terminations, truncations, infos = env.step(act)
            renderer.collect()
        renderer.render()
        print(f'observation_shape: {obs.shape}, action_shape: {act.shape}')
    env.close()


def gym_simulation_process(env, renderer, model, num_episodes):
    obs = env.reset()  # 在循环外reset一次，因为向量环境会自动地reset
    for e in range(num_episodes):
        renderer.reset()
        dones = [False]
        while not any(dones):
            act, _ = model.predict(obs)
            obs, rewards, dones, infos = env.step(act)
            renderer.collect()
        renderer.render()
    print(f'observation_shape: {obs.shape}, action_shape: {act.shape}')
    env.close()
