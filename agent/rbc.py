import numpy as np


class RBC:
    def __init__(self, AC_attributes, BESS_attributes, washer_attributes, observation_variables, t_band=1, p_active=0.2,
                 mode='test'):

        self.AC_nominal_power = AC_attributes['nominal_power']
        self.BESS_charge_nominal_power = BESS_attributes['charge_nominal_power']
        self.BESS_discharge_nominal_power = BESS_attributes['discharge_nominal_power']
        self.washer_nominal_power = washer_attributes['nominal_power']

        self.observation_variables = observation_variables
        self.t_band = t_band
        self.p_active = p_active
        self.mode = mode

        self.occupancy_index = self.observation_variables.index('occupancy')
        self.indoor_temperature_index = self.observation_variables.index('indoor_temperature')
        self.target_temperature_index = self.observation_variables.index('target_temperature')
        self.home_electrical_power_index = self.observation_variables.index('home_electrical_power')
        self.BESS_power_index = self.observation_variables.index('BESS_power')
        self.AC_state_index = self.observation_variables.index('AC_state')
        self.BESS_state_index = self.observation_variables.index('BESS_state')
        self.laundry_demand_index = self.observation_variables.index('laundry_demand')

        self.laundry_demand_met = 0

        self.action = {
            "AC": {
                "control": 1,
                "state_expectation": 0,
                "power_expectation": self.AC_nominal_power,
            },
            "BESS": {
                "control": 1,
                "state_expectation": 0,
                "power_expectation": 0,
            },
            "washer": {
                "control": 1,
                "state_expectation": 0,
                "power_expectation": self.washer_nominal_power,
            }

        }

        self.AC_count = 0
        self.BESS_count = 0
        self.washer_count = 0
        self.AC_control_history = [0]
        self.BESS_control_history = [0]
        self.washer_control_history = [0]

    def __formal(self, observation: np.ndarray):

        # AC control
        if observation[self.occupancy_index] == 0:
            self.action['AC']['state_expectation'] = 0
        elif observation[self.occupancy_index] == 1:
            if observation[self.indoor_temperature_index] < observation[self.target_temperature_index] - self.t_band:
                self.action['AC']['state_expectation'] = 1
            elif observation[self.indoor_temperature_index] > observation[self.target_temperature_index] + self.t_band:
                self.action['AC']['state_expectation'] = -1
            else:
                # self.action['AC']['state_expectation'] = observation[self.AC_state_index]
                self.action['AC']['state_expectation'] = 0
        else:
            raise ValueError("Invalid occupancy value")

        # BESS control
        no_BESS_power = observation[self.home_electrical_power_index] - observation[self.BESS_power_index]

        if no_BESS_power > self.p_active:
            self.action['BESS']['state_expectation'] = -1
            self.action['BESS']['power_expectation'] = self.BESS_discharge_nominal_power
        elif no_BESS_power < -self.p_active:
            self.action['BESS']['state_expectation'] = 1
            self.action['BESS']['power_expectation'] = self.BESS_charge_nominal_power
        else:
            # self.action['BESS']['state_expectation'] = observation[self.BESS_state_index]
            self.action['BESS']['state_expectation'] = 0

        # washer control
        if observation[self.laundry_demand_index] and self.laundry_demand_met == 0:
            self.action['washer']['state_expectation'] = 1
            self.laundry_demand_met = 1
        else:
            self.action['washer']['state_expectation'] = 0

        # calculate control count
        if observation[self.AC_state_index] != self.action['AC']['state_expectation']:
            self.AC_count += 1
            self.AC_control_history.append(1)
        else:
            self.AC_control_history.append(0)
        if observation[self.BESS_state_index] != self.action['BESS']['state_expectation']:
            self.BESS_count += 1
            self.BESS_control_history.append(1)
        else:
            self.BESS_control_history.append(0)
        return self.action

    def __test(self, observation):
        self.action['AC']['state_expectation'] = 1
        return self.action

    def predict(self, observation):

        if self.mode == 'formal':
            return self.__formal(observation)  # state is None
        elif self.mode == 'test':
            return self.__test(observation)
        else:
            raise ValueError("Invalid mode value")
