from typing import Any, Mapping, Optional


class Device:
    """
    家庭能源设备基类
    """
    default = {
        'name': 'light',
        'controllable': True,
        'nominal_power': 0.05,
        'standby_power': 0.01,
    }

    def __init__(self, name: str, controllable: bool, nominal_power: Optional[float], standby_power: Optional[float],
                 **kwargs: Any):
        # 属性
        self.name = name  # 设备名称
        self.controllable = controllable  # 是否可控
        self.nominal_power = nominal_power  # 额定功率
        self.standby_power = standby_power  # 待机功率，指设备处于待机状态时需要消耗外部电能的功率

        # 变动参数
        self.state: Optional[int] = None  # 设备状态 (0: 待机, 不为0: 运行)
        self.power: Optional[float] = None  # 实际功率

    def update(self, control_signal: dict):
        """
        更新设备状态和功率, 控制惯性：若未被控制，则保持原状态和功率，先确定状态，再确定功率
        """
        if control_signal is None:
            self.state = self.state
            signal_power = self.power
        else:
            self.state = control_signal['state_expectation']
            signal_power = control_signal['power_expectation']
        if self.state != 0:
            self.power = min(signal_power, self.nominal_power)
        else:
            self.power = self.standby_power

    def reset(self):
        self.state: int = 0
        self.power: float = 0.0

    def get_metadata(self) -> Mapping[str, Any]:
        return {
            'name': self.name,
            'controllable': self.controllable,
            'nominal_power': self.nominal_power,
            'standby_power': self.standby_power,
        }


class IndivisibleDevice(Device):
    """
    不可分割设备类
    适用洗衣机洗碗机烘干机等设备
    """
    default = {
        'name': 'washer',
        'controllable': True,
        'nominal_power': 1.20,
        'standby_power': 0.01,
        'min_continuous_working_time': 45,
        'minutes_per_time_step': 5,
    }

    def __init__(self, name: str, nominal_power: float, standby_power: float, min_continuous_working_time: float,
                 minutes_per_time_step: float, **kwargs: Any):
        super().__init__(name=name, controllable=True, nominal_power=nominal_power, standby_power=standby_power,
                         **kwargs)

        # 属性
        self.min_continuous_working_time = min_continuous_working_time  # 最小连续工作时间
        self.continuous_working_time: Optional[float] = None  # 连续工作时间
        self.minutes_per_time_step = minutes_per_time_step

    def update(self, control_signal):
        assert self.continuous_working_time is not None, 'continuous_working_time must be provided for update'

        if self.continuous_working_time < self.min_continuous_working_time and self.state != 0:
            self.state = self.state
            self.power = self.power
        else:
            super().update(control_signal)

        if self.state == 0:
            self.continuous_working_time = 0
        else:
            self.continuous_working_time += self.minutes_per_time_step

    def reset(self):
        super().reset()
        self.continuous_working_time = 0

    def get_metadata(self) -> Mapping[str, Any]:
        return {
            **super().get_metadata(),
            'min_continuous_working_time': self.min_continuous_working_time,
            'minutes_per_time_step': self.minutes_per_time_step,
        }


class HeatPump(Device):
    """
    热泵类
    适用于空调、冰箱等热泵设备
    """
    default = {
        'name': 'AC',
        'controllable': True,
        'nominal_power': 8,
        'standby_power': 0.01,
        'efficiency': 0.04,
        'max_cop': 3.91,
    }

    def __init__(self, name: str, nominal_power: float, standby_power: float,
                 efficiency: float, max_cop: float, **kwargs: Any):
        super().__init__(name=name, controllable=True, nominal_power=nominal_power, standby_power=standby_power,
                         **kwargs)
        # 属性
        self.max_cop = max_cop
        self.efficiency = efficiency

        # 变动参数
        self.cop: Optional[float] = None  # cop通过get_cop方法计算

    def __update_cop(self, indoor_temperature: float, outdoor_temperature: float):

        def c_to_k(x):
            return x + 273.15

        if self.state == 1:
            cop = self.efficiency * c_to_k(indoor_temperature) / (
                    indoor_temperature - outdoor_temperature + 1e-8)
        elif self.state == -1:
            cop = self.efficiency * c_to_k(indoor_temperature) / (
                    outdoor_temperature - indoor_temperature + 1e-8)
        elif self.state == 0:
            cop = 0
        else:
            raise ValueError('state must be one of [-1, 0, 1]')

        if cop < 0 or cop > self.max_cop:
            cop = self.max_cop
        else:
            cop = cop
        self.cop = cop

    def output_heat_power(self):
        # 制冷量\制热量 = COP * 输入功率
        if self.state == 0:
            return 0
        elif self.state == 1:
            return self.power * self.cop
        elif self.state == -1:
            return - self.power * self.cop  # 制冷量为负

    def update(self, control_signal, indoor_temperature: float = None, outdoor_temperature: float = None):
        super().update(control_signal)
        assert indoor_temperature or outdoor_temperature, 'indoor_temperature and outdoor_temperature must be provided for update'
        self.__update_cop(indoor_temperature, outdoor_temperature)

    def reset(self, indoor_temperature: float = None, outdoor_temperature: float = None):
        super().reset()
        assert indoor_temperature or outdoor_temperature, 'indoor_temperature and outdoor_temperature must be provided for reset'
        self.__update_cop(indoor_temperature, outdoor_temperature)

    def get_metadata(self) -> Mapping[str, Any]:
        return {
            **super().get_metadata(),
            'efficiency': self.efficiency,
            'max_cop': self.max_cop,
        }


class Battery(Device):
    """
    电池类
    适用于各类电池设备
    self.power >= 0，需要结合self.state判断是充电还是放电
    self.state = 1: 充电
    self.state = -1: 放电
    """
    default = {
        'name': 'BESS',
        'controllable': True,
        'charge_nominal_power': 10,
        'discharge_nominal_power': 10,
        'charge_efficiency': 0.95,
        'discharge_efficiency': 0.95,
        'capacity': 10,
        'loss_coefficient': 0.01,  # 电池自身损耗系数
        'minutes_per_time_step': 5,  # 涉及到功率和能量的转换，需要知道模拟时间步长，单位为分钟
    }

    def __init__(self, name: str, charge_nominal_power: float, discharge_nominal_power: float,
                 charge_efficiency: float, discharge_efficiency: float, capacity: float, loss_coefficient: float,
                 minutes_per_time_step: int, **kwargs: Any):
        super().__init__(name=name, controllable=True, nominal_power=None, standby_power=None, **kwargs)
        # 属性
        self.charge_nominal_power = charge_nominal_power  # 充电额定功率
        self.discharge_nominal_power = discharge_nominal_power  # 放电额定功率
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.capacity = capacity  # KwH
        self.loss_coefficient = loss_coefficient  # 一个小时会自损失的电量
        self.hours_per_time_step = minutes_per_time_step / 60

        # 变动参数
        self.soc: Optional[float] = None  # 电池的soc
        self.battery_energy: Optional[float] = None  # 电池的能量

    def __e_to_p(self, energy):
        return energy / self.hours_per_time_step  # 将能量转换为功率

    def __p_to_e(self, power):
        return power * self.hours_per_time_step  # 将功率转换为能量

    def __update_battery_energy_soc(self):
        """
        更新电池的能量和soc
        """
        energy = self.__p_to_e(self.power)

        if self.state == 1:
            self.battery_energy = self.battery_energy + energy * self.charge_efficiency
        elif self.state == -1:
            self.battery_energy = self.battery_energy - energy / self.discharge_efficiency
        elif self.state == 0:
            self.battery_energy = self.battery_energy
        else:
            raise ValueError('state must be one of [-1, 0, 1]')

        self.battery_energy = self.battery_energy - self.__p_to_e(
            self.loss_coefficient * self.battery_energy)  # 因为loss_coefficient是每小时的损耗，所以相当于是功率，所以要转换为能量

        self.soc = self.battery_energy / self.capacity

    def update(self, control_signal):
        """
        电池的功率不仅受到额定功率限制，而且还受到电池电量的限制
        """

        if control_signal is None:
            self.state = self.state
            signal_power = self.power
        else:
            self.state = control_signal['state_expectation']
            signal_power = control_signal['power_expectation']

        if self.state == 0:
            self.power = 0
        elif self.state == 1:
            self.power = min(self.__e_to_p(self.capacity - self.battery_energy) / self.charge_efficiency,
                             self.charge_nominal_power, signal_power)
        elif self.state == -1:
            self.power = min(self.__e_to_p(self.battery_energy) * self.discharge_efficiency,
                             self.discharge_nominal_power, signal_power)
        else:
            raise ValueError('state must be one of [-1, 0, 1]')

        self.__update_battery_energy_soc()

    def reset(self):
        super().reset()
        self.soc: float = 0.2
        self.battery_energy = self.soc * self.capacity

    def get_metadata(self) -> Mapping[str, Any]:
        return {
            **super().get_metadata(),
            'charge_nominal_power': self.charge_nominal_power,
            'discharge_nominal_power': self.discharge_nominal_power,
            'charge_efficiency': self.charge_efficiency,
            'discharge_efficiency': self.discharge_efficiency,
            'capacity': self.capacity,
            'loss_coefficient': self.loss_coefficient,
            'hours_per_time_step': self.hours_per_time_step,
        }


class Photovoltaic(Device):
    """
    光伏类
    适用于各类光伏设备
    """
    default = {
        'name': 'PV',
        'efficiency': 0.2,  # 光伏板效率
        'nominal_power': 1.3,  # 1平方米光伏板额定功率 单位kW
        'min_solar_radiation': 10,  # 最小发电光照强度 单位W/m^2
        'panel_area': 50,  # 光伏板面积 单位m^2
    }

    def __init__(self, name: str, nominal_power: float, efficiency: float, min_solar_radiation: float,
                 panel_area: float,
                 **kwargs: Any):
        super().__init__(name=name, controllable=False, nominal_power=nominal_power, standby_power=None, **kwargs)
        # 属性
        self.efficiency = efficiency
        self.min_solar_radiation = min_solar_radiation
        self.panel_area = panel_area

    def update(self, solar_radiation: float):
        if solar_radiation < self.min_solar_radiation:
            self.state = 0
            self.power = 0
        else:
            self.state = 1
            self.power = min(solar_radiation * self.efficiency / 1000, self.nominal_power) * self.panel_area

    def reset(self, solar_radiation: float = None):
        assert solar_radiation is not None, 'solar_radiation must be provided for reset'
        self.update(solar_radiation)

    def get_metadata(self) -> Mapping[str, Any]:
        return {
            **super().get_metadata(),
            'efficiency': self.efficiency,
            'min_solar_radiation': self.min_solar_radiation,
            'panel_area': self.panel_area,
        }
