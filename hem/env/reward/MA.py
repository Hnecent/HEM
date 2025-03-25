import math
import numpy as np


def ac_reward(state: dict):
    indoor_temperature = state['indoor_temperature_history'][-1]
    target_temperature = state['target_temperature_history'][-1]
    occupancy = state['occupancy_history'][-1]

    # Set hyperparameters
    temperature_band = 0  # temperature band
    # temperature comfort reward
    comfort_reward = occupancy * -math.pow(min(0.0, temperature_band - abs(indoor_temperature - target_temperature)), 2)

    return comfort_reward
    # return 2


def washer_reward(state: dict):
    # laundry demand reward
    delayed_duration = state['delayed_duration_history'][-1]
    laundry_demand_reward = math.exp(-delayed_duration)

    # cost reward
    cost = state['home_electrical_cost_history'][-1]
    cost_reward = math.exp(-cost)
    # cost_reward = - cost

    laundry_demand_weight = 1
    cost_weight = 1

    return laundry_demand_reward * laundry_demand_weight + cost_reward * cost_weight


def bess_reward(state: dict):
    home_electrical_power = state['home_electrical_power_history'][-1]
    home_no_BESS_electrical_power = state['home_no_BESS_electrical_power_history'][-1]
    BESS_state = state['BESS_state_history'][-1]
    BESS_power = state['BESS_power_history'][-1]
    base_load = state['base_load_history'][-1]

    # energy storage reward
    # energy_storage_reward = math.exp(-abs(home_electrical_power)) * (
    #         -np.sign(home_no_BESS_electrical_power) * BESS_state) * BESS_power

    # energy_storage_reward = math.exp(-abs(home_electrical_power))

    energy_storage_reward = math.exp(-abs(home_electrical_power - base_load))

    # # energy_storage_reward = -state['home_electrical_cost_history'][-1]
    #
    # energy_storage_reward = -math.pow((BESS_power - state['base_load_history'][-1]), 2)
    #

    return energy_storage_reward
    # return 2
