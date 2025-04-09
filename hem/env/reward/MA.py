import math
import numpy as np
import yaml
import os

config_path = os.path.join(os.path.dirname(__file__), '../config_env.yaml')
config = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))


def ac_reward(state: dict):
    indoor_temperature = state['indoor_temperature_history'][-1]
    target_temperature = state['target_temperature_history'][-1]
    occupancy = state['occupancy_history'][-1]
    AC_control = state['AC_control_history'][-1]
    AC_power = state['AC_power_history'][-1]
    AC_previous_power = state['AC_power_history'][-2]

    # temperature comfort reward
    # Set hyperparameters
    temperature_band = 0  # temperature band
    comfort_reward = occupancy * -math.pow(min(0.0, temperature_band - abs(indoor_temperature - target_temperature)), 2)

    # control efficiency reward
    control_efficiency_reward = -AC_control * math.exp(
        -abs(AC_power - AC_previous_power) / config['AC_ATTRIBUTES']['nominal_power'])

    # weights
    comfort_weight = 1.5
    control_efficiency_weight = 1

    return comfort_reward * comfort_weight + control_efficiency_reward * control_efficiency_weight


def washer_reward(state: dict):
    # laundry demand reward
    delayed_duration = state['delayed_duration_history'][-1]
    cost = state['home_electrical_cost_history'][-1]
    washer_control = state['washer_control_history'][-1]
    washer_power = state['washer_power_history'][-1]
    washer_previous_power = state['washer_power_history'][-2]

    # laundry demand reward
    laundry_demand_reward = math.exp(-delayed_duration)

    # cost reward
    # cost_reward = math.exp(-cost)
    cost_reward = - cost

    # control efficiency reward
    control_efficiency_reward = -washer_control * math.exp(
        -abs(washer_power - washer_previous_power) / config['WASHER_ATTRIBUTES']['nominal_power'])

    # weights
    laundry_demand_weight = 6
    cost_weight = 6
    control_efficiency_weight = 1

    return laundry_demand_reward * laundry_demand_weight + cost_reward * cost_weight + control_efficiency_reward * control_efficiency_weight


def bess_reward(state: dict):
    home_electrical_power = state['home_electrical_power_history'][-1]
    home_no_BESS_electrical_power = state['home_no_BESS_electrical_power_history'][-1]
    BESS_control = state['BESS_control_history'][-1]
    BESS_state = state['BESS_state_history'][-1]
    BESS_power = state['BESS_power_history'][-1]
    base_load = state['base_load_history'][-1]
    price = state['electrical_price_history'][-1]
    BESS_previous_power = state['BESS_power_history'][-2]

    # energy storage reward
    energy_storage_reward = math.exp(-abs(home_electrical_power - base_load)) * 4

    # control efficiency reward
    control_efficiency_reward = -BESS_control * math.exp(
        -abs(BESS_power - BESS_previous_power) / config['BESS_ATTRIBUTES']['charge_nominal_power'])

    # weights
    energy_storage_weight = 5
    control_efficiency_weight = 1

    return energy_storage_reward * energy_storage_weight + control_efficiency_reward * control_efficiency_weight
