import gymnasium
import functools
import numpy as np
import random
import yaml
import copy

from hem.env.objects.device import Battery, HeatPump, Photovoltaic, IndivisibleDevice
from hem.env.objects.base import EpisodeTracker
from hem.env.objects.data import Data
from hem.env.objects.dynamics import AirHeatDynamics
from hem.env.objects.demand import Demand
import hem.env.ami as ami
from hem.env.ami import AMI
from hem.env.reward import MA

from pettingzoo.utils.env import AgentID, ObsType
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils import agent_selector, wrappers

import supersuit as ss
from supersuit.multiagent_wrappers.padding_wrappers import pad_action_space_v0


def aec_env(**kwargs):
    env = RawAECEnv(**kwargs)
    env = wrappers.BaseWrapper(env)
    env = pad_action_space_v0(env)
    return env


def parallel_env(**kwargs):
    env = RawParallelEnv(**kwargs)
    env = pad_action_space_v0(env)
    return env


def sa_env(**kwargs):
    env = RawSAEnv(**kwargs)
    env = pad_action_space_v0(env)
    env = ss.normalize_obs_v0(env)
    env = ss.frame_stack_v2(env, stack_size=5)
    return env


def gym_env(**kwargs):
    env = RawGymEnv(**kwargs)
    env = ss.normalize_obs_v0(env)
    env = ss.frame_stack_v2(env, stack_size=5)
    return env


def sa_stack_env(**kwargs):
    env = RawSAEnv(**kwargs)
    env = pad_action_space_v0(env)
    env = ss.frame_stack_v2(env, stack_size=5)
    env = ss.normalize_obs_v0(env)
    return env


class HEMEnv:
    """
    HEMEnv is the main class for the Home Energy Management Environment.
    Parameters
    ----------
    random_episode_split: bool
        Whether to randomly split episodes.
    rolling_episode_split: bool
        Whether to split episodes in a rolling manner.
    data_start_end: tuple[int, int]
        Start and end indices of the data. In this case, the data is a year of one minute-level data.
    episode_time_steps: int
        Number of time steps in an episode.
    mode: str
        Mode of the simulation, one of 'train', 'val', 'test', or 'whole'.
    random_seed: int
        Random seed for reproducibility.
    noise_strength: float
        Strength of the noise added to the data.
    config_path: str
        Path to the configuration file.
    """

    def __init__(self, random_episode_split: bool, rolling_episode_split: bool, data_start_end: tuple[int, int],
                 episode_time_steps: int, mode: str, random_seed: int, noise_strength: float, config_path: str):
        # parameters
        self.random_episode_split = random_episode_split
        self.rolling_episode_split = rolling_episode_split
        self.data_start_end = data_start_end
        self.episode_time_steps = episode_time_steps
        self.mode = mode
        self.random_seed = random_seed
        self.noise_strength = noise_strength
        random.seed(self.random_seed)

        # load config
        self.config = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
        self.MINUTES_PER_TIME_STEP = self.config['MINUTES_PER_TIME_STEP']
        # data objects
        self.data = Data(data_start_end, self.MINUTES_PER_TIME_STEP, mode, random_seed, noise_strength)
        self.data4pre_base_load = Data(data_start_end, ami.PRE_USED_MINUTES_PER_TIME_STEP, mode, random_seed,
                                       noise_strength)

        # devices objects
        self.AC = HeatPump(**self.config['AC_ATTRIBUTES'])
        self.BESS = Battery(**self.config['BESS_ATTRIBUTES'])
        self.PV = Photovoltaic(**self.config['PV_ATTRIBUTES'])
        self.washer = IndivisibleDevice(**self.config['WASHER_ATTRIBUTES'])

        # demand
        self.demand = Demand(self.data, self.washer, self.random_seed)

        # air heat dynamic model
        self.air_heat_dynamic = AirHeatDynamics(**self.config['AIR_HEAT_DYNAMICS_ATTRIBUTES'])

        # AMI
        self.AMI = AMI(self.MINUTES_PER_TIME_STEP, self.data, self.data4pre_base_load, self.AC, self.BESS, self.PV,
                       self.washer, self.demand, self.air_heat_dynamic)

        # time step
        self.time_step = None

        # init action
        self.init_action = {'AC': {'control': 1, 'state_expectation': 0, 'power_expectation': 0},
                            'washer': {'control': 1, 'state_expectation': 0, 'power_expectation': 0},
                            'BESS': {'control': 1, 'state_expectation': 0, 'power_expectation': 0}}

        # episode tracker
        self.episode_tracker = EpisodeTracker(0, self.data.length)

        self.metadata = self.get_metadata()

    def _action_safe_layer(self, action: dict):

        """
        save energy rules
        """
        # AC
        if self.AMI.running_buffer['occupancy_history'][-1] == 0:
            action['AC']['state_expectation'] = 0

        # washer
        if self.AMI.running_buffer['laundry_demand_history'][-1] == 0:
            action['washer']['state_expectation'] = 0

        """
        Assistance rules
        """
        # BESS base_load 辅助动作
        if action['BESS']['state_expectation'] != 0:
            action['BESS']['power_expectation'] = max(action['BESS']['power_expectation'] - action['BESS'][
                'state_expectation'] * self.AMI.running_buffer['base_load_pre_history'][-1], 0)
        else:
            action['BESS']['state_expectation'] = -1
            action['BESS']['power_expectation'] = self.AMI.running_buffer['base_load_pre_history'][-1]

        """
        on_demand rules
        """
        # AC
        if action['AC']['state_expectation'] == self.AMI.running_buffer['AC_state_history'][-1] and abs(
                action['AC']['power_expectation'] - self.AMI.running_buffer['AC_power_history'][-1]) < 0.05:
            action['AC']['control'] = 0

        # BESS
        if action['BESS']['state_expectation'] == self.AMI.running_buffer['BESS_state_history'][-1] and abs(
                action['BESS']['power_expectation'] - self.AMI.running_buffer['BESS_power_history'][-1]) < 0.05:
            action['BESS']['control'] = 0

        # washer
        if action['washer']['state_expectation'] == self.AMI.running_buffer['washer_state_history'][-1]:
            action['washer']['control'] = 0

        return action

    def step(self, action: dict):
        assert self.time_step is not None, 'Please reset the environment before taking a step.'

        action = self._action_safe_layer(action)

        #  从action中获取控制信号
        AC_action = action["AC"]
        BESS_action = action["BESS"]
        washer_action = action["washer"]

        AC_control_signal = self.AMI.control(AC_action)
        BESS_control_signal = self.AMI.control(BESS_action)
        washer_control_signal = self.AMI.control(washer_action)

        #  更新设备状态
        #  重要：当未执行collect_running_data时，running_buffer中的数据是上一个时间步的数据
        #  空调使用上一个时间步的室内温度和室外温度更新COP，并根据动作计算输出功率
        self.AC.update(AC_control_signal, self.AMI.running_buffer['indoor_temperature_history'][-1],
                       self.AMI.running_buffer['outdoor_temperature_history'][-1])
        #  电池本身存储的是上一个时间步的电量
        self.BESS.update(BESS_control_signal)
        self.washer.update(washer_control_signal)

        #  由于PV转换的瞬时性，使用当前的太阳辐射更新PV的输出功率
        self.PV.update(self.AMI.data.solar_radiation[self.time_step])

        #  更新空气热动力学模型
        self.air_heat_dynamic.update(self.AMI.running_buffer['outdoor_temperature_history'][-1],
                                     self.AMI.running_buffer['solar_radiation_history'][-1],
                                     self.AC.output_heat_power())

        #  更新需求
        self.demand.update(self.time_step)

        #  使用AMI收集数据（内含load_balance和carbon_production）
        self.AMI.collect_running_data(action, self.time_step)

        self.time_step += 1

    def reset(self):
        self.episode_tracker.next_episode(self.episode_time_steps, self.rolling_episode_split,
                                          self.random_episode_split, random_seed=self.random_seed)
        self.time_step = self.episode_tracker.episode_start_time_step

        self.demand.reset(self.time_step)

        self.AC.reset(indoor_temperature=self.demand.target_temperature,
                      outdoor_temperature=self.AMI.data.outdoor_temperature[self.time_step])
        self.BESS.reset()
        self.washer.reset()
        self.PV.reset(solar_radiation=self.AMI.data.solar_radiation[self.time_step])

        self.air_heat_dynamic.reset(init_indoor_temperature=self.demand.target_temperature)

        self.AMI.reset()

        self.AMI.collect_running_data(self.init_action, self.time_step)

    def get_metadata(self):
        return {
            'minutes_per_time_step': self.MINUTES_PER_TIME_STEP,
            'random_episode_split': self.random_episode_split,
            'rolling_episode_split': self.rolling_episode_split,
            'data_start_end': f'{self.AMI.data.length} in {self.data_start_end}',
            'episode_time_steps': self.episode_time_steps,
            'mode': self.mode,
            'random_seed': self.random_seed,
            'noise_strength': self.noise_strength,
            'AC_attributes': self.AC.get_metadata(),
            'BESS_attributes': self.BESS.get_metadata(),
            'PV_attributes': self.PV.get_metadata(),
            'air_heat_dynamics_attributes': self.air_heat_dynamic.get_metadata(),
            'observation': self.AMI.observation_variables,
        }


class RawAECEnv(AECEnv):
    metadata = {'name': 'RawAECEnv_for_HEM', 'is_parallelizable': False, 'multi_agent': True}

    def __init__(self, random_episode_split: bool, rolling_episode_split: bool, data_start_end: tuple[int, int],
                 episode_time_steps: int, mode: str, random_seed: int, noise_strength: float, config_path: str,
                 render_mode: str):
        super().__init__()
        self.hem_env = HEMEnv(random_episode_split, rolling_episode_split, data_start_end, episode_time_steps, mode,
                              random_seed, noise_strength, config_path)
        self.state = self.hem_env.AMI.running_buffer  # state是环境的所有内部状态

        self.possible_agents = ['AC', 'washer', 'BESS']

        self._action_spaces = {'AC': gymnasium.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
                               'washer': gymnasium.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
                               'BESS': gymnasium.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32), }

        self._observation_space = {
            agent: gymnasium.spaces.Box(low=-1e5, high=1e5, shape=(len(self.hem_env.AMI.observation_variables),),
                                        dtype=np.float32) for agent in self.possible_agents}

        self.reward_map = {'AC': MA.ac_reward, 'washer': MA.washer_reward, 'BESS': MA.bess_reward}
        self.reward_history = {agent: [] for agent in self.possible_agents}

        self.render_mode = render_mode
        assert self.render_mode == 'None', 'Only None render_mode is supported.'

    def reset(self, seed=None, options=None):
        self.hem_env.reset()

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.reward_history = {agent: [0] for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {'NA': "NA"} for agent in self.agents}
        self.state = self.hem_env.AMI.running_buffer
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

    def step(self, actions):
        if (
                self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(actions)
            return

        agent = self.agent_selection

        # 清除当前 agent 的累积奖励和奖励
        self._cumulative_rewards[agent] = 0
        self._clear_rewards()

        # 转换动作
        actions = self._action_transform(agent, actions)

        # 执行环境步进
        self.hem_env.step(actions)
        self.state = self.hem_env.AMI.running_buffer

        # 计算奖励并赋值给当前 agent
        self.rewards[agent] = self.reward_map[agent](self.state)
        self.reward_history[agent].append(self.rewards[agent])

        # 累积奖励到 _cumulative_rewards
        self._accumulate_rewards()

        # 更新终止/截断状态
        self.terminations[agent] = (
                self.hem_env.time_step == self.hem_env.episode_tracker.episode_end_time_step
        )

        # 切换到下一个 agent
        self.agent_selection = self._agent_selector.next()

    def observe(self, agent: AgentID) -> ObsType | None:
        agent_observation = self.hem_env.AMI.observe()
        return agent_observation

    def render(self):
        raise NotImplementedError('External rendering using a renderer.')

    def close(self):
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self._observation_space[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self._action_spaces[agent]

    def _action_transform(self, agent: AgentID, actions: np.ndarray):
        control = np.argmax(actions[0:2])

        if agent == 'AC':
            state_expectation = np.argmax(actions[2:5]) - 1
            power_expectation = (actions[5] + 1) * self.hem_env.AC.nominal_power / 2
        elif agent == 'washer':
            state_expectation = np.argmax(actions[2:4])
            power_expectation = self.hem_env.washer.nominal_power
        elif agent == 'BESS':
            state_expectation = np.argmax(actions[2:5]) - 1
            if state_expectation > 0:
                power_expectation = (actions[5] + 1) * self.hem_env.BESS.charge_nominal_power / 2
            else:
                power_expectation = (actions[5] + 1) * self.hem_env.BESS.discharge_nominal_power / 2
        else:
            raise ValueError(f'Unknown agent {agent}')

        actions_template = copy.deepcopy(self.hem_env.init_action)

        actions_template[agent].update({'control': control, 'state_expectation': state_expectation,
                                        'power_expectation': power_expectation})

        return actions_template


class RawParallelEnv(ParallelEnv):
    metadata = {'name': 'RawParallelEnv_for_HEM', 'multi_agent': True}

    def __init__(self, random_episode_split: bool, rolling_episode_split: bool, data_start_end: tuple[int, int],
                 episode_time_steps: int, mode: str, random_seed: int, noise_strength: float, config_path: str,
                 render_mode: str):
        super().__init__()
        self.hem_env = HEMEnv(random_episode_split, rolling_episode_split, data_start_end, episode_time_steps, mode,
                              random_seed, noise_strength, config_path)
        self.state = self.hem_env.AMI.running_buffer  # state是环境的所有内部状态

        self.possible_agents = ['AC', 'washer', 'BESS']

        self._action_spaces = {'AC': gymnasium.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
                               'washer': gymnasium.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
                               'BESS': gymnasium.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32), }

        self._observation_space = {
            agent: gymnasium.spaces.Box(low=-1e5, high=1e5, shape=(len(self.hem_env.AMI.observation_variables),),
                                        dtype=np.float32) for agent in self.possible_agents}

        self.reward_map = {'AC': MA.ac_reward, 'washer': MA.washer_reward, 'BESS': MA.bess_reward}
        self.reward_history = {agent: [] for agent in self.possible_agents}

        self.render_mode = render_mode
        assert self.render_mode == 'None', 'Only None render_mode is supported.'

    def reset(self, seed=None, options=None):
        self.hem_env.reset()

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.reward_history = {agent: [0] for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {'NA': "NA"} for agent in self.agents}
        self.state = self.hem_env.AMI.running_buffer
        self.observation = {agent: self.observe(agent) for agent in self.agents}

        return self.observation, self.infos

    def step(self, actions):

        # 转换动作
        actions = {agent: self._action_transform(agent, actions[agent]) for agent in self.agents}

        # 执行环境步进
        self.hem_env.step(actions)
        self.state = self.hem_env.AMI.running_buffer

        # 计算奖励并赋值给当前 agent
        for agent in self.agents:
            self.rewards[agent] = self.reward_map[agent](self.state)
            self.reward_history[agent].append(self.rewards[agent])

        # 更新终止/截断状态
        for agent in self.agents:
            self.terminations[agent] = (self.hem_env.time_step == self.hem_env.episode_tracker.episode_end_time_step)

        self.observation = {agent: self.observe(agent) for agent in self.agents}

        return self.observation, self.rewards, self.terminations, self.truncations, self.infos

    def observe(self, agent: AgentID) -> ObsType | None:
        agent_observation = self.hem_env.AMI.observe()
        return agent_observation

    def render(self):
        raise NotImplementedError('External rendering using a renderer.')

    def close(self):
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self._observation_space[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self._action_spaces[agent]

    def _action_transform(self, agent: AgentID, actions: np.ndarray):
        control = np.argmax(actions[0:2])

        if agent == 'AC':
            state_expectation = np.argmax(actions[2:5]) - 1
            power_expectation = (actions[5] + 1) * self.hem_env.AC.nominal_power / 2
        elif agent == 'washer':
            state_expectation = np.argmax(actions[2:4])
            power_expectation = self.hem_env.washer.nominal_power
        elif agent == 'BESS':
            state_expectation = np.argmax(actions[2:5]) - 1
            if state_expectation > 0:
                power_expectation = (actions[5] + 1) * self.hem_env.BESS.charge_nominal_power / 2
            else:
                power_expectation = (actions[5] + 1) * self.hem_env.BESS.discharge_nominal_power / 2
        else:
            raise ValueError(f'Unknown agent {agent}')

        actions_template = copy.deepcopy(self.hem_env.init_action)

        actions_template[agent].update({'control': control, 'state_expectation': state_expectation,
                                        'power_expectation': power_expectation})

        return actions_template[agent]


class RawSAEnv(ParallelEnv):
    metadata = {'name': 'RawSAEnv_for_HEM', 'black_death': True, 'multi_agent': True}

    def __init__(self, random_episode_split: bool, rolling_episode_split: bool, data_start_end: tuple[int, int],
                 episode_time_steps: int, mode: str, random_seed: int, noise_strength: float, config_path: str,
                 render_mode: str):
        super().__init__()
        self.hem_env = HEMEnv(random_episode_split, rolling_episode_split, data_start_end, episode_time_steps, mode,
                              random_seed, noise_strength, config_path)
        self.state = self.hem_env.AMI.running_buffer  # state是环境的所有内部状态

        self.possible_agents = ['washer']

        self._action_spaces = {'AC': gymnasium.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
                               'washer': gymnasium.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
                               'BESS': gymnasium.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32), }

        self._observation_space = {
            agent: gymnasium.spaces.Box(low=-1e5, high=1e5, shape=(len(self.hem_env.AMI.observation_variables),),
                                        dtype=np.float32) for agent in self.possible_agents}

        self.reward_map = {'AC': MA.ac_reward, 'washer': MA.washer_reward, 'BESS': MA.bess_reward}

        self.reward_render_agents = ['AC', 'washer', 'BESS']
        self.reward_history = {agent: [] for agent in self.reward_render_agents}

        self.render_mode = render_mode
        assert self.render_mode == 'None', 'Only None render_mode is supported.'

    def reset(self, seed=None, options=None):
        self.hem_env.reset()

        self.agents = self.possible_agents[:]  # AC
        self.rewards = {agent: 0 for agent in self.agents}
        self.reward_history = {agent: [0] for agent in self.reward_render_agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {'NA': "NA"} for agent in self.agents}
        self.state = self.hem_env.AMI.running_buffer
        self.observation = {agent: self.observe(agent) for agent in self.agents}

        return self.observation, self.infos

    def step(self, actions):

        agent = self.agents[0]

        # 转换动作
        actions = self._action_transform(agent, actions[agent])

        # 执行环境步进
        self.hem_env.step(actions)
        self.state = self.hem_env.AMI.running_buffer

        # 计算奖励并赋值给当前 agent
        for agent in self.agents:
            self.rewards[agent] = self.reward_map[agent](self.state)
            self.reward_history[agent].append(self.rewards[agent])

        # 更新终止/截断状态
        for agent in self.agents:
            self.terminations[agent] = (self.hem_env.time_step == self.hem_env.episode_tracker.episode_end_time_step)

        self.observation = {agent: self.observe(agent) for agent in self.agents}

        # print(f'step: {self.observation, self.hem_env.time_step}')

        return self.observation, self.rewards, self.terminations, self.truncations, self.infos

    def observe(self, agent: AgentID) -> ObsType | None:
        agent_observation = self.hem_env.AMI.observe()
        return agent_observation

    def render(self):
        raise NotImplementedError('External rendering using a renderer.')

    def close(self):
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self._observation_space[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self._action_spaces[agent]

    def _action_transform(self, agent: AgentID, actions: np.ndarray):
        control = np.argmax(actions[0:2])

        if agent == 'AC':
            state_expectation = np.argmax(actions[2:5]) - 1
            power_expectation = (actions[5] + 1) * self.hem_env.AC.nominal_power / 2
        elif agent == 'washer':
            state_expectation = np.argmax(actions[2:4])
            power_expectation = self.hem_env.washer.nominal_power
        elif agent == 'BESS':
            state_expectation = np.argmax(actions[2:5]) - 1
            if state_expectation > 0:
                power_expectation = (actions[5] + 1) * self.hem_env.BESS.charge_nominal_power / 2
            else:
                power_expectation = (actions[5] + 1) * self.hem_env.BESS.discharge_nominal_power / 2
        else:
            raise ValueError(f'Unknown agent {agent}')

        actions_template = copy.deepcopy(self.hem_env.init_action)

        actions_template[agent].update({'control': control, 'state_expectation': state_expectation,
                                        'power_expectation': power_expectation})

        return actions_template


class RawGymEnv(gymnasium.Env):
    metadata = {'name': 'RawGymEnv_for_HEM', 'multi_agent': False}

    def __init__(self, random_episode_split: bool, rolling_episode_split: bool, data_start_end: tuple[int, int],
                 episode_time_steps: int, mode: str, random_seed: int, noise_strength: float, config_path: str,
                 render_mode: str):
        super().__init__()
        self.hem_env = HEMEnv(random_episode_split, rolling_episode_split, data_start_end, episode_time_steps, mode,
                              random_seed, noise_strength, config_path)
        self.state = self.hem_env.AMI.running_buffer
        self.observation_space = gymnasium.spaces.Box(low=-1e5, high=1e5,
                                                      shape=(len(self.hem_env.AMI.observation_variables),),
                                                      dtype=np.float32)
        self.action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(6 + 4 + 6,), dtype=np.float32)

        self.possible_agents = ['AC', 'washer', 'BESS']
        self.agents = self.possible_agents
        self.reward_map = {'AC': MA.ac_reward, 'washer': MA.washer_reward, 'BESS': MA.bess_reward}
        self.reward_history = {agent: [] for agent in self.possible_agents}

        self.render_mode = render_mode
        assert self.render_mode == 'None', 'Only None render_mode is supported.'

    def reset(self, seed=None, options=None):
        self.hem_env.reset()
        self.state = self.hem_env.AMI.running_buffer
        self.rewards = {agent: 0 for agent in self.agents}
        self.reward_history = {agent: [0] for agent in self.possible_agents}
        self.terminations = False
        self.truncations = False
        self.infos = {agent: {'NA': "NA"} for agent in self.agents}

        return self.observe(), self.infos

    def step(self, actions):
        actions = self._action_transform(actions)
        self.hem_env.step(actions)
        self.state = self.hem_env.AMI.running_buffer

        for agent in self.agents:
            self.rewards[agent] = self.reward_map[agent](self.state)
            self.reward_history[agent].append(self.rewards[agent])

        self.terminations = (self.hem_env.time_step == self.hem_env.episode_tracker.episode_end_time_step)

        return self.observe(), self.cal_rewards(), self.terminations, self.truncations, self.infos

    def observe(self) -> ObsType | None:
        observation = self.hem_env.AMI.observe()
        return observation

    def cal_rewards(self) -> float:
        reward_fl = self.rewards['washer'] + self.rewards['BESS'] + self.rewards['AC']
        return reward_fl

    def _action_transform(self, actions: np.ndarray):

        AC_control = np.argmax(actions[0:2])
        AC_state_expectation = np.argmax(actions[2:5]) - 1
        AC_power_expectation = (actions[5] + 1) * self.hem_env.AC.nominal_power / 2
        BESS_control = np.argmax(actions[6:8])
        BESS_state_expectation = np.argmax(actions[8:11]) - 1
        if BESS_state_expectation > 0:
            BESS_power_expectation = (actions[11] + 1) * self.hem_env.BESS.charge_nominal_power / 2
        else:
            BESS_power_expectation = (actions[11] + 1) * self.hem_env.BESS.discharge_nominal_power / 2
        washer_control = np.argmax(actions[12:14])
        washer_state_expectation = int(np.argmax(actions[14:16]))
        washer_power_expectation = self.hem_env.washer.nominal_power

        actions_template = copy.deepcopy(self.hem_env.init_action)

        actions_template['AC'].update({'control': AC_control, 'state_expectation': AC_state_expectation,
                                       'power_expectation': AC_power_expectation})
        actions_template['BESS'].update({'control': BESS_control, 'state_expectation': BESS_state_expectation,
                                         'power_expectation': BESS_power_expectation})
        actions_template['washer'].update({'control': washer_control, 'state_expectation': washer_state_expectation,
                                           'power_expectation': washer_power_expectation})

        return actions_template
