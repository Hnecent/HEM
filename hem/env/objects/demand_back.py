import random
import numpy as np
from hem.env.objects.device import IndivisibleDevice
from hem.env.objects.data import Data


class Demand:
    """
    Used to generate users demand for energy equipment
    """

    def __init__(self, data: Data, washer: IndivisibleDevice, random_seed: int):

        self.data = data
        self.washer = washer
        random.seed(random_seed)
        np.random.seed(random_seed)

        self.target_temp_params = {"mean": 22, "std": 2, "min": 18, "max": 28}

        self.laundry_demand_bins = {
            "night": (0, 5, 0.01),  # 0:00-5:00，概率1%
            "morning": (5, 8, 0.1),  # 5:00-8:00，概率10%
            "day": (8, 17, 0.3),  # 8:00-17:00，概率30%（假设有人在家）
            "evening": (17, 20, 0.6),  # 17:00-20:00，概率60%
            "late_night": (20, 24, 0.02)  # 20:00-24:00，概率2%
        }

        self.target_temperature = None
        self.laundry_demand = None

    def update(self, time_step: int):
        self.__update_target_temperature()
        self.__update_laundry_demand(time_step)

    def __update_target_temperature(self):
        self.target_temperature = self.target_temperature

    def __update_laundry_demand(self, time_step: int):
        """
        模拟一天的洗衣机启动时间点
        """
        self.laundry_demand = 0
        # 获取当前时间段和占用状态
        current_hour = self.data.hour[time_step]
        is_occupied = self.data.occupancy[time_step]
        washer_state = self.washer.state

        # 获取当前时间段概率
        base_prob = 0.0
        for (start_h, end_h, prob) in self.laundry_demand_bins.values():
            if start_h <= current_hour < end_h:
                base_prob = prob * 10
                break

        # 计算实际概率
        if is_occupied == 1 and washer_state != 1:
            prob = base_prob  # 可进一步叠加其他因素（如周末）
        else:
            prob = 0

        # 随机触发
        if np.random.rand() < prob / 60:  # 每分钟概率 = 每小时概率 / 60
            self.laundry_demand = 1

    def reset(self):
        temp = random.gauss(self.target_temp_params["mean"], self.target_temp_params["std"])
        self.target_temperature = np.clip(temp, self.target_temp_params["min"], self.target_temp_params["max"])
        self.laundry_demand = 0
