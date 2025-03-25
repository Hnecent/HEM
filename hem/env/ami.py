import numpy as np
import torch
import torch.nn as nn
import os

from hem.env.objects.data import Data
from hem.env.objects.device import HeatPump, Battery, Photovoltaic, IndivisibleDevice
from hem.env.objects.demand import Demand
from hem.env.objects.dynamics import AirHeatDynamics

PRE_BASE_LOAD_MODEL_PATH = os.path.join(os.path.dirname(__file__), './base_load_model/best_model.pth')
PRE_BASE_LOAD_SCALER_PATH = os.path.join(os.path.dirname(__file__), './base_load_model/scaler.npz')
PRE_USED_TIME_STEPS = 24  # 使用过去24个时间点预测下一个点
PRE_USED_MINUTES_PER_TIME_STEP = 5


class AMI:
    """
    Advanced Metering Infrastructure
    1. 通过控制策略，对可控家庭能源设备进行控制
    2. 采集运行数据，获取家庭能源设备的状态和功率
    """

    def __init__(self, minutes_per_time_step: int, data: Data, AC: HeatPump, BESS: Battery, PV: Photovoltaic,
                 washer: IndivisibleDevice, demand: Demand, air_heat_dynamics: AirHeatDynamics, pre: bool = True,
                 real_pre: bool = False):

        self.minutes_per_time_step = minutes_per_time_step  # 控制时间步长
        self.data = data  # data object
        self.AC = AC
        self.BESS = BESS
        self.PV = PV
        self.washer = washer
        self.demand = demand
        self.air_heat_dynamics = air_heat_dynamics

        self.__init_buffer()
        self.laundry_demand_time = None
        self.laundry_allowed_border = None
        self.laundry_demand_count = 0
        self.washer_start_time = None
        self.washer_start_count = 0

        # base_load prediction model
        self.base_load_pre_model = TimeSeriesModel()

        self.observation_variables = [
            # AC
            'AC_control', 'AC_state', 'AC_power',
            # BESS
            'BESS_control', 'BESS_state', 'BESS_power', 'BESS_soc', 'BESS_energy',
            # washer
            'washer_control', 'washer_state', 'washer_power', 'washer_continuous_working_time',
            # PV
            'PV_power',
            # load balance
            'home_electrical_power', 'home_electrical_consumption', 'home_electrical_cost',
            # carbon
            'carbon_intensity', 'carbon_production',  # carbon
            # demand
            'target_temperature', 'laundry_demand', 'laundry_allowed_waiting_time',
            # heat dynamic model
            'indoor_temperature',
            # time
            'hour_of_day', 'minute_of_hour', 'day_of_week',
            # weather
            'solar_radiation', 'outdoor_temperature',
            # electrical & occupancy
            'electrical_price',
            'base_load',
            'occupancy',
            # auxiliary
            'delayed_duration',
        ]
        # pre
        if pre:

            if real_pre:
                self.base_load_pre_model.load_state_dict(torch.load(PRE_BASE_LOAD_MODEL_PATH))
                self.scaler = np.load(PRE_BASE_LOAD_SCALER_PATH, allow_pickle=True)['scaler'].item()

            self.observation_variables += [
                'base_load_pre',

                'solar_radiation_pre',
                'outdoor_temperature_pre',
                'electrical_price_pre',
                'occupancy_pre'
            ]

        self.real_pre = real_pre

    def __init_buffer(self):
        self.running_buffer = {

            # AC
            'AC_control_history': [],
            'AC_state_expectation_history': [],
            'AC_power_expectation_history': [],
            'AC_state_history': [],
            'AC_power_history': [],
            'AC_cop_history': [],

            # BESS
            'BESS_control_history': [],
            'BESS_state_expectation_history': [],
            'BESS_power_expectation_history': [],
            'BESS_state_history': [],
            'BESS_power_history': [],
            'BESS_soc_history': [],
            'BESS_energy_history': [],

            # washer
            'washer_control_history': [],
            'washer_state_expectation_history': [],
            'washer_power_expectation_history': [],
            'washer_state_history': [],
            'washer_power_history': [],
            'washer_continuous_working_time_history': [],

            # PV
            'PV_state_history': [],
            'PV_power_history': [],

            # load balance
            'home_electrical_power_history': [],
            'home_electrical_consumption_history': [],
            'home_electrical_cost_history': [],
            'home_no_BESS_electrical_power_history': [],  # no BESS
            'home_no_BESS_electrical_consumption_history': [],  # no BESS
            'home_no_BESS_electrical_cost_history': [],  # no BESS

            # carbon
            'carbon_intensity_history': [],
            'carbon_production_history': [],
            'no_BESS_carbon_production_history': [],  # no BESS

            # demand
            'target_temperature_history': [],
            'laundry_demand_history': [],
            'laundry_allowed_waiting_time_history': [],

            # heat dynamic model
            'indoor_temperature_history': [],

            # time
            'month_of_year_history': [],
            'day_of_month_history': [],
            'hour_of_day_history': [],
            'minute_of_hour_history': [],
            'day_of_week_history': [],

            # weather
            'solar_radiation_history': [],
            'outdoor_temperature_history': [],

            # electrical & occupancy
            'electrical_price_history': [],
            'base_load_history': [],
            'occupancy_history': [],

            # auxiliary
            'delayed_duration_history': [],

            # pre
            'base_load_pre_history': [],

            'solar_radiation_pre_history': [],
            'outdoor_temperature_pre_history': [],
            'electrical_price_pre_history': [],
            'occupancy_pre_history': []
        }

    @staticmethod
    def control(device_action: dict):
        if device_action['control'] == 0:
            return None
        elif device_action['control'] == 1:
            return {'state_expectation': device_action['state_expectation'],
                    'power_expectation': device_action['power_expectation']}
        else:
            raise ValueError('control must be one of [0, 1]')

    def load_balance(self, time_step: int):

        power = self.AC.power + self.BESS.state * self.BESS.power + self.washer.power - self.PV.power + \
                self.data.base_load[time_step]
        consumption = max(power * self.minutes_per_time_step / 60, 0)
        cost = consumption * self.data.electrical_price[time_step]

        no_BESS_power = power - self.BESS.state * self.BESS.power
        no_BESS_consumption = max(no_BESS_power * self.minutes_per_time_step / 60, 0)
        no_BESS_cost = no_BESS_consumption * self.data.electrical_price[time_step]
        return power, consumption, cost, no_BESS_power, no_BESS_consumption, no_BESS_cost

    def carbon_production(self, time_step: int):

        carbon_production = self.running_buffer['home_electrical_consumption_history'][-1] * self.data.carbon_intensity[
            time_step]
        no_BESS_carbon_production = self.running_buffer['home_no_BESS_electrical_consumption_history'][-1] * \
                                    self.data.carbon_intensity[time_step]
        return carbon_production, no_BESS_carbon_production

    def delayed_duration(self, time_step: int):

        """
        计算启动洗衣机的延迟时间，即洗衣机启动时间与（洗衣需求时间 + 最大允许等待时间）的差值

        """

        if self.running_buffer['laundry_demand_history'][-1] == 1 and self.running_buffer['laundry_demand_history'][
            -2] == 0:
            self.laundry_demand_time = time_step
            self.laundry_allowed_border = time_step + (
                    self.running_buffer['laundry_allowed_waiting_time_history'][-1] / self.minutes_per_time_step)
            self.laundry_demand_count += 1

        if self.running_buffer['washer_state_history'][-1] == 1 and self.running_buffer['washer_state_history'][
            -2] and (self.laundry_demand_count > self.washer_start_count):
            self.washer_start_time = time_step
            self.washer_start_count += 1

        if self.laundry_demand_count <= self.washer_start_count:
            delayed_duration = 0
        else:
            delayed_duration = max(time_step - self.laundry_allowed_border, 0)

        return delayed_duration

    def pre_base_load(self, time_step: int):

        if self.real_pre:
            self.base_load_pre_model.eval()

            # 使用前5个时间步的数据预测下一个时间步的数据，不够的用0填充
            # 准备输入数据
            sequence_length = PRE_USED_TIME_STEPS
            input_features = []

            # 收集历史数据（自动填充不足部分）
            for t in range(time_step - sequence_length + 1, time_step + 1):
                if t < 0:  # 数据不足时用0填充
                    features = [
                        0.0,  # base_load
                        0.0,  # month
                        0.0,  # day
                        0.0,  # hour
                        0.0,  # minute
                        0.0  # day_of_week
                    ]
                else:
                    features = [
                        self.data.base_load[t],
                        self.data.month[t],
                        self.data.day[t],
                        self.data.hour[t],
                        self.data.minute[t],
                        self.data.day_type[t]
                    ]
                input_features.append(features)
            # 转换为numpy数组
            input_array = np.array(input_features, dtype=np.float32)
            # 数据标准化（使用训练时的scaler）
            scaled_input = np.zeros_like(input_array)
            feature_names = ['base_load', 'month', 'day', 'hour', 'minute', 'day_of_week']
            for i, name in enumerate(feature_names):
                scaler = self.scaler[name]
                scaled_input[:, i] = scaler.transform(input_array[:, i].reshape(-1, 1)).flatten()
            # 转换为PyTorch Tensor
            input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0)  # (1, USE_PRE_TIME_STEP, 6)
            # 执行预测
            with torch.no_grad():
                prediction = self.base_load_pre_model(input_tensor).cpu().numpy().flatten()[0]
            # 反标准化预测结果
            base_load_scaler = self.scaler['base_load']
            base_load_pre = base_load_scaler.inverse_transform([[prediction]])[0][0]
        else:
            try:
                base_load_pre = self.data.base_load[time_step + 1]
            except KeyError:
                base_load_pre = self.data.base_load[time_step]

        return base_load_pre

    def collect_running_data(self, action: dict, time_step: int):
        """
        collect the running data of the environment
        """
        # AC
        self.running_buffer['AC_control_history'].append(action['AC']['control'])
        self.running_buffer['AC_state_expectation_history'].append(action['AC']['state_expectation'])
        self.running_buffer['AC_power_expectation_history'].append(action['AC']['power_expectation'])
        self.running_buffer['AC_state_history'].append(self.AC.state)
        self.running_buffer['AC_power_history'].append(self.AC.power)
        self.running_buffer['AC_cop_history'].append(self.AC.cop)

        # BESS
        self.running_buffer['BESS_control_history'].append(action['BESS']['control'])
        self.running_buffer['BESS_state_expectation_history'].append(action['BESS']['state_expectation'])
        self.running_buffer['BESS_power_expectation_history'].append(action['BESS']['power_expectation'])
        self.running_buffer['BESS_state_history'].append(self.BESS.state)
        self.running_buffer['BESS_power_history'].append(self.BESS.power)
        self.running_buffer['BESS_soc_history'].append(self.BESS.soc)
        self.running_buffer['BESS_energy_history'].append(self.BESS.battery_energy)

        # washer
        self.running_buffer['washer_control_history'].append(action['washer']['control'])
        self.running_buffer['washer_state_expectation_history'].append(action['washer']['state_expectation'])
        self.running_buffer['washer_power_expectation_history'].append(action['washer']['power_expectation'])
        self.running_buffer['washer_state_history'].append(self.washer.state)
        self.running_buffer['washer_power_history'].append(self.washer.power)
        self.running_buffer['washer_continuous_working_time_history'].append(self.washer.continuous_working_time)

        # PV
        self.running_buffer['PV_state_history'].append(self.PV.state)
        self.running_buffer['PV_power_history'].append(self.PV.power)

        # load balance
        power, consumption, cost, no_BESS_power, no_BESS_consumption, no_BESS_cost = self.load_balance(time_step)
        self.running_buffer['home_electrical_power_history'].append(power)
        self.running_buffer['home_electrical_consumption_history'].append(consumption)
        self.running_buffer['home_electrical_cost_history'].append(cost)
        self.running_buffer['home_no_BESS_electrical_power_history'].append(no_BESS_power)  # no BESS
        self.running_buffer['home_no_BESS_electrical_consumption_history'].append(no_BESS_consumption)  # no BESS
        self.running_buffer['home_no_BESS_electrical_cost_history'].append(no_BESS_cost)  # no BESS

        # carbon
        carbon_production, no_BESS_carbon_production = self.carbon_production(time_step)
        self.running_buffer['carbon_intensity_history'].append(self.data.carbon_intensity[time_step])
        self.running_buffer['carbon_production_history'].append(carbon_production)
        self.running_buffer['no_BESS_carbon_production_history'].append(no_BESS_carbon_production)  # no BESS

        # demand
        self.running_buffer['target_temperature_history'].append(self.demand.target_temperature)
        self.running_buffer['laundry_demand_history'].append(self.demand.laundry_demand)
        self.running_buffer['laundry_allowed_waiting_time_history'].append(self.demand.laundry_allowed_waiting_time)

        # heat dynamic model
        self.running_buffer['indoor_temperature_history'].append(self.air_heat_dynamics.indoor_temperature)

        # time
        self.running_buffer['month_of_year_history'].append(self.data.month[time_step])
        self.running_buffer['day_of_month_history'].append(self.data.day[time_step])
        self.running_buffer['hour_of_day_history'].append(self.data.hour[time_step])
        self.running_buffer['minute_of_hour_history'].append(self.data.minute[time_step])
        self.running_buffer['day_of_week_history'].append(self.data.day_type[time_step])

        # weather
        self.running_buffer['solar_radiation_history'].append(self.data.solar_radiation[time_step])
        self.running_buffer['outdoor_temperature_history'].append(self.data.outdoor_temperature[time_step])

        # electrical & occupancy
        self.running_buffer['electrical_price_history'].append(self.data.electrical_price[time_step])
        self.running_buffer['base_load_history'].append(self.data.base_load[time_step])
        self.running_buffer['occupancy_history'].append(self.data.occupancy[time_step])

        # auxiliary
        self.running_buffer['delayed_duration_history'].append(self.delayed_duration(time_step))

        # pre
        self.running_buffer['base_load_pre_history'].append(self.pre_base_load(time_step))
        try:
            self.running_buffer['solar_radiation_pre_history'].append(self.data.solar_radiation[time_step + 1])
            self.running_buffer['outdoor_temperature_pre_history'].append(self.data.outdoor_temperature[time_step + 1])
            self.running_buffer['electrical_price_pre_history'].append(self.data.electrical_price[time_step + 1])
            self.running_buffer['occupancy_pre_history'].append(self.data.occupancy[time_step + 1])
        except KeyError:
            self.running_buffer['solar_radiation_pre_history'].append(self.data.solar_radiation[time_step])
            self.running_buffer['outdoor_temperature_pre_history'].append(self.data.outdoor_temperature[time_step])
            self.running_buffer['electrical_price_pre_history'].append(self.data.electrical_price[time_step])
            self.running_buffer['occupancy_pre_history'].append(self.data.occupancy[time_step])

    def observe(self) -> np.ndarray:
        """
        return the observation of the environment
        """
        observation = np.array(
            [self.running_buffer[variable + '_history'][-1] for variable in self.observation_variables],
            dtype=np.float32)
        return observation

    def reset(self):
        self.__init_buffer()
        self.laundry_demand_time = None
        self.laundry_allowed_border = None
        self.laundry_demand_count = 0
        self.washer_start_time = None
        self.washer_start_count = 0


class TimeSeriesModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2):
        super(TimeSeriesModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)  # out shape: (batch_size, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# class TimeSeriesModel(nn.Module):
#     def __init__(self, input_size=6, hidden_size=64, num_layers=2):
#         super().__init__()
#         # 使用GRU替代LSTM（参数更少）
#         self.gru = nn.GRU(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=0.1
#         )
#
#         # 时间注意力机制（轻量化设计）
#         self.attention = nn.Sequential(
#             nn.Linear(hidden_size, 16),
#             nn.Tanh(),
#             nn.Linear(16, 1),
#             nn.Softmax(dim=1)
#         )
#
#         # 深度可分离卷积提取局部特征（参数量减少3-4倍）
#         self.conv = nn.Sequential(
#             nn.Conv1d(hidden_size, hidden_size, 3, padding=1, groups=hidden_size),
#             nn.BatchNorm1d(hidden_size),
#             nn.GELU()
#         )
#
#         # 预测头
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, 32),
#             nn.LayerNorm(32),
#             nn.GELU(),
#             nn.Linear(32, 1)
#         )
#
#     def forward(self, x):
#         out, _ = self.gru(x)  # [batch, seq_len, hidden]
#
#         # 时间注意力加权
#         attn_weights = self.attention(out)  # [batch, seq_len, 1]
#         context = torch.sum(attn_weights * out, dim=1)  # [batch, hidden]
#
#         # 卷积处理
#         context = self.conv(context.unsqueeze(-1)).squeeze()  # [batch, hidden]
#
#         return self.fc(context)


# class TimeSeriesModel(nn.Module):
#     def __init__(self, input_size=6, hidden_size=64, num_layers=2):
#         super().__init__()
#         # 双向GRU替代LSTM（参数更少且防止过拟合）
#         self.gru = nn.GRU(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=True,
#             dropout=0.3  # 增加dropout比例
#         )
#
#         # 时序注意力机制（轻量版）
#         self.attention = nn.Sequential(
#             nn.Linear(2 * hidden_size, 16),  # 双向输出维度是2*hidden
#             nn.Tanh(),
#             nn.LayerNorm(16),
#             nn.Linear(16, 1),
#             nn.Softmax(dim=1)
#         )
#
#         # 正则化预测头
#         self.fc = nn.Sequential(
#             nn.Linear(2 * hidden_size, 32),
#             nn.Dropout(0.2),
#             nn.LayerNorm(32),
#             nn.GELU(),  # 更平滑的激活函数
#             nn.Linear(32, 1)
#         )
#
#     def forward(self, x):
#         # GRU层
#         gru_out, _ = self.gru(x)  # [batch, seq_len, 2*hidden]
#
#         # 注意力加权
#         attn_weights = self.attention(gru_out)  # [batch, seq_len, 1]
#         context = torch.sum(attn_weights * gru_out, dim=1)  # [batch, 2*hidden]
#
#         # 正则化预测
#         return self.fc(context)
