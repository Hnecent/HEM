import numpy as np
import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/california_3938_2016_minute_data_modify_carbon.csv')


def get_mean_std(data_line):
    return np.mean(data_line), np.std(data_line)


class Data:
    def __init__(self, data_start_end: tuple, minutes_per_time_step: int,
                 mode: str, random_seed: int, noise_strength: float):
        np.random.seed(random_seed)
        # read data and determine time interval
        original_data = pd.read_csv(DATA_PATH)
        data = original_data[data_start_end[0]:data_start_end[1]:minutes_per_time_step]
        data.reset_index(drop=True, inplace=True)

        # split data
        val_idx = {
            'Month': [6, 7, 8, 9],
            'Day': [1, 10, 20, 30],
            # 'Day': [],
        }
        test_idx = {
            'Month': [6, 7, 8, 9],
            'Day': [1, 10, 20, 30],
            # 'Day': [],
        }
        val_data = data[(data['Month'].isin(val_idx['Month'])) & (data['Day'].isin(val_idx['Day']))]
        test_data = data[(data['Month'].isin(test_idx['Month'])) & (data['Day'].isin(test_idx['Day']))]
        train_data = data[~data.index.isin(val_data.index) & ~data.index.isin(test_data.index)]

        print(f'Make env: mode: {mode}, '
              f'whole_days: {int(len(data) * minutes_per_time_step / 1440)}, '
              f'train_days: {int(len(train_data) * minutes_per_time_step / 1440)}, '
              f'val_days: {int(len(val_data) * minutes_per_time_step / 1440)}, '
              f'test_days: {int(len(test_data) * minutes_per_time_step / 1440)}')

        if mode == 'val':
            data = val_data
        elif mode == 'test':
            data = test_data
        elif mode == 'train':
            data = train_data
        elif mode == 'whole':
            data = data
        else:
            raise ValueError('mode must be one of "train", "val", "test" or "whole"')

        data.reset_index(drop=True, inplace=True)

        # original data
        self.year = data['Year']

        self.month = data['Month'].astype(int)
        self.day = data['Day'].astype(int)
        self.hour = data['Hour'].astype(int)
        self.minute = data['Minute'].astype(int)
        self.day_type = data['Day Type'].astype(int)
        self.solar_radiation = data['Downwelling Global Solar [W/m2]'].astype(np.float32)
        self.outdoor_temperature = data['Outdoor Drybulb Temperature [C]'].astype(np.float32)
        self.base_load = data['Equipment Electric Power [kW]'].astype(np.float32)
        self.electrical_price = data['Electricity Pricing [$/kWh]'].astype(np.float32) * 100  # cent
        self.occupancy = data['Occupancy'].astype(int)
        self.carbon_intensity = data['Carbon Intensity [kg/kWh]'].astype(np.float32)

        self.length = len(data)

        # add noise
        self.solar_radiation_mean, self.solar_radiation_std = get_mean_std(
            original_data['Downwelling Global Solar [W/m2]'])
        self.outdoor_temperature_mean, self.outdoor_temperature_std = get_mean_std(
            original_data['Outdoor Drybulb Temperature [C]'])
        self.base_load_mean, self.base_load_std = get_mean_std(original_data['Equipment Electric Power [kW]'])

        self.solar_radiation += np.random.normal(0, noise_strength * self.solar_radiation_std, self.length)
        self.outdoor_temperature += np.random.normal(0, noise_strength * self.outdoor_temperature_std, self.length)
        self.base_load += np.random.normal(0, noise_strength * self.base_load_std, self.length)
