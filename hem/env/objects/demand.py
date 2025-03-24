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
        self.laundry_demand = None

        # laundry demand
        self.laundry_demand_bins = {
            1: 0.6,  # Monday
            2: 0.8,  # Tuesday
            3: 0.8,  # Wednesday
            4: 0.8,  # Thursday
            5: 0.8,  # Friday
            6: 0.8,  # Saturday
            7: 0.8  # Sunday
        }

        self.episode_laundry_demand = None

    def update(self, time_step: int):
        self.__update_target_temperature()
        self.__update_laundry_demand(time_step)

    def __update_target_temperature(self):
        self.target_temperature = self.episode_target_temperature

    def __update_laundry_demand(self, time_step: int):
        pass

    def reset(self):
        temp = random.gauss(self.target_temp_params["mean"], self.target_temp_params["std"])
        self.episode_target_temperature = np.clip(temp, self.target_temp_params["min"], self.target_temp_params["max"])

        episode_probs = [self.laundry_demand_bins[self.data.day[0]]]
