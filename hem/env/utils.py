import matplotlib.pyplot as plt
from hem.env.core import RawAECEnv, ParallelEnv, RawSAEnv
from hem.env.kpis import CostFunction
import numpy as np
from collections import deque
import torch.nn.functional as F
import torch


class Renderer:
    def __init__(self, render_mode: str, env: RawAECEnv | ParallelEnv | RawSAEnv):
        """
        Args:
            render_mode (str): The mode of rendering. One of 'step', 'episode', 'none'.
            env (RawAECEnv | ParallelEnv): The unwrapped environment to render.
        """

        self.render_mode = render_mode

        self.env = env
        self.state_buffer = deque(maxlen=2)
        self.reward_history_buffer = deque(maxlen=2)
        self.episode_time_buffer = deque(maxlen=2)
        self.state = None
        self.reward_history = None
        self.episode_time = (None, None)

        # 添加以下属性用于保存图表对象
        self.fig = None
        self.axes = None
        self.line = None

        possible_render_modes = ['step', 'episode', 'none']
        if self.render_mode not in possible_render_modes:
            raise ValueError(f'render_mode must be one of {possible_render_modes}')

    def collect(self):
        self.state_buffer.append(self.env.state)
        self.reward_history_buffer.append(self.env.reward_history)
        self.episode_time = (
            self.env.hem_env.episode_tracker.episode_start_time_step,
            self.env.hem_env.episode_tracker.episode_end_time_step)
        self.episode_time_buffer.append(self.episode_time)

    def _step(self):
        # 如果是首次渲染，初始化图表
        if self.fig is None:
            plt.ion()  # 开启交互模式
            self.fig, self.axes = plt.subplots()
            (self.line,) = self.axes.plot(
                self.state['indoor_temperature_history'],
                'b-',  # 蓝色实线
                label='Indoor Temperature'
            )
            self.axes.set_xlabel('Time Step')
            self.axes.set_ylabel('Temperature (°C)')
            self.axes.legend()
            plt.show()
        else:
            # 更新折线图数据
            x_data = range(len(self.state['indoor_temperature_history']))
            self.line.set_data(x_data, self.state['indoor_temperature_history'])
            # 调整坐标轴范围
            self.axes.relim()
            self.axes.autoscale_view()
            # 强制刷新画布
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        # 短暂暂停让图表更新（控制刷新频率）
        plt.pause(0.001)

    def _episode(self):
        # 如果是首次渲染，初始化图表
        exclude_list = ['AC_control', 'BESS_control', 'washer_control', 'BESS_electrical_consumption_contribution',
                        'BESS_electrical_cost_contribution', 'washer_delayed_duration']

        print('Episode Time:', self.episode_time)
        print('KPIs:')
        for k, v in self._cal_kpis().items():
            if k in exclude_list:
                print(f'{k}: {v}')
            else:
                print(f'{k}: {v:.4f}')

        fig, axes = plt.subplots(8, 2)
        fig.set_size_inches(14, 16)

        """
        AC 相关
        """
        axes[0, 0].stairs(self.state['outdoor_temperature_history'], label='outdoor_temperature')
        axes[0, 0].stairs(self.state['indoor_temperature_history'], label='indoor_temperature')
        axes[0, 0].stairs(self.state['target_temperature_history'], label='target_temperature')
        axes[0, 0].legend()

        # AC控制和状态
        axes[1, 0].stairs(self.state['AC_control_history'], label='AC_control')
        axes[1, 0].stairs(self.state['AC_state_history'], label='AC_state')
        axes[1, 0].set_ylim(-1.15, 1.15)
        axes[1, 0].legend()

        #  AC功率和动作预期功率
        axes[2, 0].stairs(self.state['AC_power_expectation_history'], label='AC_power_expectation', alpha=0.5)
        axes[2, 0].stairs(self.state['AC_power_history'], label='AC_power', alpha=0.5)
        axes[2, 0].legend()

        color = 'tab:blue'
        axes[3, 0].stairs(
            np.array(self.state['indoor_temperature_history']) - np.array(
                self.state['target_temperature_history']),
            label='difference of indoor_target', color=color)
        axes[3, 0].plot([1] * len(self.state['indoor_temperature_history']), label='1',
                        color='black', linestyle='--')
        axes[3, 0].plot([-1] * len(self.state['indoor_temperature_history']), label='-1',
                        color='black', linestyle='--')
        axes[3, 0].tick_params(axis='y', labelcolor=color)
        axes[3, 0].set_ylabel('difference of indoor_target', color=color)

        ax = axes[3, 0].twinx()
        color = 'tab:red'
        ax.stairs(self.state['occupancy_history'], label='occupancy', color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_ylabel('occupancy', color=color)

        # AC reward
        axes[4, 0].stairs(self.reward_history['AC'], label='AC_reward', color='orange')
        axes[4, 0].legend()

        """
        BESS相关
        """
        # 设备功率
        axes[0, 1].stairs(self.state['AC_power_history'], label='AC_power')
        axes[0, 1].stairs(np.array(self.state['BESS_power_history']) * np.array(self.state['BESS_state_history']),
                          label='BESS_power')
        axes[0, 1].stairs([-x for x in self.state['PV_power_history']], label='PV_power')
        axes[0, 1].stairs(self.state['base_load_history'], label='base_load')
        axes[0, 1].stairs(self.state['washer_power_history'], label='washer_power')
        axes[0, 1].legend(ncol=3)

        # BESS控制和状态
        axes[1, 1].stairs(self.state['BESS_control_history'], label='BESS_control')
        axes[1, 1].stairs(self.state['BESS_state_history'], label='BESS_state')
        axes[1, 1].set_ylim(-1.15, 1.15)
        axes[1, 1].legend()

        # BESS功率和动作预期功率和电价
        axes[2, 1].stairs(self.state['BESS_power_expectation_history'], label='BESS_power_expectation', alpha=0.5)
        axes[2, 1].stairs(self.state['BESS_power_history'], label='BESS_power', alpha=0.5)
        axes[2, 1].legend()

        ax = axes[2, 1].twinx()
        color = 'tab:red'
        ax.stairs(self.state['electrical_price_history'], label='electrical_price', color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_ylabel('electrical_price', color=color)

        # 家庭电力功率和无BESS功率
        axes[3, 1].stairs(self.state['home_electrical_power_history'], label='home_electrical_power', alpha=0.5)
        axes[3, 1].stairs(self.state['home_no_BESS_electrical_power_history'], label='no_BESS_power', alpha=0.5)
        axes[3, 1].plot([0] * len(self.state['home_no_BESS_electrical_power_history']), label='0', color='black',
                        linestyle='--')
        axes[3, 1].legend()

        # BESS reward
        axes[4, 1].stairs(self.reward_history['BESS'], label='BESS_reward', color='orange')
        axes[4, 1].legend()

        # BESS能量和电价
        color = 'tab:blue'
        axes[5, 1].stairs(self.state['BESS_energy_history'], label='BESS_energy', color=color)
        axes[5, 1].set_ylim(-1, self.env.hem_env.BESS.capacity + 1)
        axes[5, 1].tick_params(axis='y', labelcolor=color)
        axes[5, 1].set_ylabel('BESS_energy', color=color)

        """
        washer相关
        """
        # 洗衣机控制和demand
        axes[5, 0].stairs(self.state['washer_control_history'], label='washer_control')
        axes[5, 0].stairs(self.state['washer_state_history'], label='washer_state')
        axes[5, 0].stairs(self.state['laundry_demand_history'], label='laundry_demand_history', color='red')
        axes[5, 0].set_ylim(-0.15, 1.15)
        axes[5, 0].legend()

        # 洗衣机功率
        axes[6, 0].stairs(self.state['washer_power_expectation_history'], label='washer_power_expectation',
                          alpha=0.5)
        axes[6, 0].stairs(self.state['washer_power_history'], label='washer_power', alpha=0.5)
        axes[6, 0].stairs(self.state['laundry_demand_history'], label='laundry_demand_history')
        axes[6, 0].stairs(self.state['PV_power_history'], label='PV_power')
        axes[6, 0].legend()

        ax = axes[6, 0].twinx()
        color = 'tab:red'
        ax.stairs(self.state['electrical_price_history'], label='electrical_price', color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_ylabel('electrical_price', color=color)

        # washer reward
        axes[7, 0].stairs(self.reward_history['washer'], label='washer_reward', color='red')
        axes[7, 0].legend()

        ax = axes[7, 0].twinx()
        ax.stairs(self.state['delayed_duration_history'], label='delayed_duration')
        ax.stairs(self.state['laundry_allowed_waiting_time_history'], label='laundry_allowed_waiting_time')
        ax.legend()

        """
        预测base_load
        """
        axes[6, 1].stairs(self.state['base_load_history'], label='base_load')
        axes[6, 1].stairs(self.state['base_load_pre_history'], label='base_load_pre')
        axes[6, 1].legend()

        plt.show()
        plt.close(fig)

    def _none(self):
        pass

    def reset(self):
        self.state_buffer.clear()
        self.reward_history_buffer.clear()
        self.episode_time_buffer.clear()
        self.state = None
        self.reward_history = None
        self.episode_time = (None, None)

    def render(self):
        assert len(self.state_buffer) > 0, 'Please call collect() before render()'  # 确保已经收集了数据

        self.state = self.state_buffer[-1]
        self.reward_history = self.reward_history_buffer[-1]
        self.episode_time = self.episode_time_buffer[-1]
        if len(list(self.state.values())[0]) == 1:
            self.state = self.state_buffer[-2]
            self.reward_history = self.reward_history_buffer[-2]
            self.episode_time = self.episode_time_buffer[-2]

        if self.render_mode == 'step':
            self._step()
        elif self.render_mode == 'episode':
            self._episode()
        elif self.render_mode == 'none':
            self._none()
        else:
            pass

    def _cal_kpis(self):
        """
        calculate the KPIs of the environment
        """
        BESS_energy = (self.state['BESS_energy_history'][0] - self.state['BESS_energy_history'][-1]) * \
                      self.env.hem_env.config['BESS_ATTRIBUTES']['discharge_efficiency']

        electrical_consumption_sum = CostFunction.electrical_consumption(
            self.state['home_electrical_consumption_history']) + BESS_energy

        electrical_cost_sum = CostFunction.electrical_cost(self.state['home_electrical_cost_history']) + BESS_energy * (
                0.05113 + 0.03025) / 2 * 100

        discomfort_proportion, discomfort_cold_proportion, discomfort_hot_proportion = CostFunction.discomfort(
            self.state['indoor_temperature_history'], self.state['target_temperature_history'],
            band=2, occupancy=self.state['occupancy_history'])

        # carbon_production_sum = sum(self.state['carbon_production_history']) + BESS_energy * 0.17513

        no_BESS_electrical_consumption_sum = CostFunction.electrical_consumption(
            self.state['home_no_BESS_electrical_consumption_history'])
        BESS_electrical_consumption_contribution = ((no_BESS_electrical_consumption_sum - electrical_consumption_sum)
                                                    / no_BESS_electrical_consumption_sum)

        no_BESS_electrical_cost_sum = CostFunction.electrical_cost(
            self.state['home_no_BESS_electrical_cost_history'])
        BESS_electrical_cost_contribution = ((no_BESS_electrical_cost_sum - electrical_cost_sum)
                                             / no_BESS_electrical_cost_sum)

        # no_BESS_carbon_production_sum = sum(self.state['no_BESS_carbon_production_history'])
        # BESS_carbon_production_contribution = ((no_BESS_carbon_production_sum - carbon_production_sum)
        #                                        / no_BESS_carbon_production_sum)

        AC_control = sum(self.state['AC_control_history'])
        BESS_control = sum(self.state['BESS_control_history'])
        washer_control = sum(self.state['washer_control_history'])

        AC_reward = sum(self.reward_history['AC'])
        BESS_reward = sum(self.reward_history['BESS'])
        washer_reward = sum(self.reward_history['washer'])

        delayed_duration = sum(self.state['delayed_duration_history']) * self.env.hem_env.MINUTES_PER_TIME_STEP

        base_load_pre_mse = F.mse_loss(torch.tensor(self.state['base_load_history'][1:]),
                                       torch.tensor(self.state['base_load_pre_history'][:-1])).item()

        return {
            'electrical_consumption_sum': electrical_consumption_sum,
            'electrical_cost_sum': electrical_cost_sum,
            'discomfort_proportion': discomfort_proportion,
            'discomfort_cold_proportion': discomfort_cold_proportion,
            'discomfort_hot_proportion': discomfort_hot_proportion,
            # 'carbon_production_sum': carbon_production_sum,
            'BESS_electrical_consumption_contribution': f'{electrical_consumption_sum:.4f} / {no_BESS_electrical_consumption_sum:.4f}, {BESS_electrical_consumption_contribution:.4f}',
            'BESS_electrical_cost_contribution': f'{electrical_cost_sum:.4f} / {no_BESS_electrical_cost_sum:.4f}, {BESS_electrical_cost_contribution:.4f}',
            # 'BESS_carbon_production_contribution': BESS_carbon_production_contribution
            'AC_control': f'{AC_control} / {len(self.state["AC_control_history"])}, {AC_control / len(self.state["AC_control_history"]):.4f}',
            'BESS_control': f'{BESS_control} / {len(self.state["BESS_control_history"])}, {BESS_control / len(self.state["BESS_control_history"]):.4f}',
            'washer_control': f'{washer_control} / {len(self.state["washer_control_history"])}, {washer_control / len(self.state["washer_control_history"]):.4f}',
            'AC_reward': AC_reward,
            'BESS_reward': BESS_reward,
            'washer_reward': washer_reward,
            'washer_delayed_duration': delayed_duration,
            'base_load_pre_mse': base_load_pre_mse,
        }
