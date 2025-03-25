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

        # target temperature
        self.target_temp_params = {"mean": 22, "std": 2, "min": 18, "max": 28}

        self.episode_target_temperature = None
        self.target_temperature = None

        # laundry demand
        self.laundry_demand = None
        self.laundry_allowed_waiting_time = None

    def update(self, time_step: int):
        self.__update_target_temperature()
        self.__update_laundry_demand(time_step)

    def __update_target_temperature(self):
        self.target_temperature = self.episode_target_temperature

    def __update_laundry_demand(self, time_step: int):
        self.laundry_demand = self.data.laundry_demand[time_step]
        self.laundry_allowed_waiting_time = self.data.laundry_allowed_waiting_time[time_step]

    def reset(self, time_step: int):
        temp = random.gauss(self.target_temp_params["mean"], self.target_temp_params["std"])
        self.episode_target_temperature = np.clip(temp, self.target_temp_params["min"], self.target_temp_params["max"])
        self.update(time_step)
