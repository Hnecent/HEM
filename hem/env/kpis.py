from typing import List, Union
import numpy as np
import pandas as pd


class CostFunction:
    @staticmethod
    def ramping(electrical_consumption_history: List[float], down_ramp: bool = None, net_export: bool = None) -> float:
        r"""Rolling sum of absolute difference in net electric consumption between consecutive time steps.

        Parameters
        ----------
        electrical_consumption_history: List[float]
            Electricity consumption time series.

        down_ramp: bool
            Include cases where there is reduction in consumption between consecutive time steps
            in the summation if `True`, otherwise set ramp value to zero for such cases.

        net_export: bool
            Include cases where net electric consumption is negative (net export) in the summation
             if `True`, otherwise set ramp value to zero for such cases.

        Returns
        -------
        ramping : float
            Ramping cost.

        Notes
        -----
        math::
            \textrm{ramping} = \sum_{i=1}^{n}{\lvert E_i - E_{i-1} \rvert}

        Where :math:`E_i` is the :math:`i^{\textrm{th}}` element in `electrical_consumption_history`, :math:`E`, that has a length of :math:`n`.
        """

        down_ramp = False if down_ramp is None else down_ramp
        net_export = True if net_export is None else net_export
        data = pd.DataFrame({'electrical_consumption_history': electrical_consumption_history})
        data['ramping'] = data['electrical_consumption_history'] - data['electrical_consumption_history'].shift(1)

        if down_ramp:
            data['ramping'] = data['ramping'].abs()
        else:
            data['ramping'] = data['ramping'].clip(lower=0)

        if not net_export:
            data.loc[data['electrical_consumption_history'] < 0, 'ramping'] = 0
        else:
            pass
        return data['ramping'].sum()

    @staticmethod
    def one_minus_load_factor(electrical_consumption_history: List[float]) -> float:
        r"""Difference between 1 and the load factor i.e., ratio of rolling mean demand
        to rolling peak demand over a specified period.
        越大波动越大

        Parameters
        ----------
        electrical_consumption_history : List[float]
            Electricity consumption time series.

        Returns
        -------
        1 - load_factor : float
            1 - load factor cost.
        """

        data = pd.DataFrame({'electrical_consumption_history': electrical_consumption_history})
        one_minus_load_factor = 1 - data['electrical_consumption_history'].mean() / data[
            'electrical_consumption_history'].max()
        return one_minus_load_factor

    @staticmethod
    def electrical_consumption(electrical_consumption_history: List[float] | np.ndarray) -> float:
        r"""Rolling sum of positive electrical consumption.

        It is the sum of electrical that is consumed from the grid.

        Parameters
        ----------
        electrical_consumption_history : List[float]
            Electricity consumption time series.

        Returns
        -------
        electrical_consumption: float
            Electricity consumption.
        """

        data = pd.DataFrame({'electrical_consumption_history': np.array(electrical_consumption_history).clip(min=0)})
        return data['electrical_consumption_history'].sum()

    @staticmethod
    def zero_net_energy(electrical_consumption_history: List[float]) -> float:
        r"""Rolling sum of net electrical consumption.

        It is the net sum of electricity that is consumed from the grid and self-generated from renewable sources.
        This calculation of zero net energy does not consider TDV and all time steps are weighted equally.

        Parameters
        ----------
        electrical_consumption_history : List[float]
            Electricity consumption time series.

        Returns
        -------
        zero_net_energy : float
            Zero net energy.
        """

        data = pd.DataFrame({'electrical_consumption_history': np.array(electrical_consumption_history)})
        return data['electrical_consumption_history'].sum()

    @staticmethod
    def electrical_cost(electrical_cost_history: List[float] | np.ndarray) -> float:
        r"""sum of electrical monetary cost.

        Parameters
        ----------
        electrical_cost_history : List[float]
            Cost time series.

        Returns
        -------
        cost : float
            Cost of electrical.
        """

        data = pd.DataFrame({'electrical_cost': np.array(electrical_cost_history).clip(min=0)})
        return data['electrical_cost'].sum()

    @staticmethod
    def discomfort(indoor_temperature_history: List[float], target_temperature_history: List[float],
                   band: Union[float, List[float]] = None, occupancy: List[int] = None) -> tuple[
        float, float, float]:
        r"""proportion of discomfort (total, too cold, and too hot) time steps as well as rolling minimum, maximum and average temperature delta.

        Parameters
        ----------
        indoor_temperature_history: List[float]
            Average building dry bulb temperature time series.
        target_temperature_history: List[float]
            Building thermostat set point time series.
        band: Union[float, List[float]], optional
            Comfort band above and below target_temperature_history beyond
            which occupant is assumed to be uncomfortable. The default value is
            a constant value time series of 2.0.
        occupancy: List[float], optional
            Occupant count time series. If provided, the comfort cost is
            evaluated for occupied time steps only.

        Returns
        -------
        discomfort_proportion : float
            proportion of occupied time steps where the condition
            (target_temperature_history - band) <= indoor_temperature_history <= (target_temperature_history + band) is not met.
        discomfort_cold_proportion : float
            proportion of occupied time steps where the condition indoor_temperature_history < (target_temperature_history - band) is met.
        discomfort_hot_proportion : float
            proportion of occupied time steps where the condition indoor_temperature_history > (target_temperature_history + band) is met.
        """

        # unmet times
        data = pd.DataFrame({
            'indoor_temperature_history': indoor_temperature_history,
            'target_temperature_history': target_temperature_history,
            'occupancy': [1] * len(indoor_temperature_history) if occupancy is None else occupancy
        })
        default_band = 2.0
        data['band'] = default_band if band is None else band
        occupied_time_step_count = data[data['occupancy'] > 0.0].shape[0]
        data['delta'] = data['indoor_temperature_history'] - data['target_temperature_history']
        data.loc[data['occupancy'] == 0.0, 'delta'] = 0.0

        data['discomfort'] = 0
        data.loc[data['delta'].abs() > data['band'], 'discomfort'] = 1
        discomfort_proportion = data['discomfort'].sum() / occupied_time_step_count

        # too cold
        data['discomfort_cold'] = 0
        data.loc[data['delta'] < -data['band'], 'discomfort_cold'] = 1
        discomfort_cold_proportion = data['discomfort_cold'].sum() / occupied_time_step_count

        # too hot
        data['discomfort_hot'] = 0
        data.loc[data['delta'] > data['band'], 'discomfort_hot'] = 1
        discomfort_hot_proportion = data['discomfort_hot'].sum() / occupied_time_step_count

        return discomfort_proportion, discomfort_cold_proportion, discomfort_hot_proportion

    @staticmethod
    def control_proportion(device_control_history: List[float]) -> float:
        r"""proportion of time steps where control signal is non-zero.

        Parameters
        ----------
        device_control_history : List[float]
            Control signal time series.

        Returns
        -------
        control_proportion : float
            proportion of time steps where control signal is non-zero.
        """

        data = pd.DataFrame({'control': np.array(device_control_history)})
        return data['control'].sum() / len(data['control'])

    @staticmethod
    def electrical_cost(electrical_cost_history: List[float]) -> float:
        r"""sum of electrical monetary cost.

        Parameters
        ----------
        electrical_cost_history : List[float]
            Cost time series.

        Returns
        -------
        cost : float
            Cost of electrical.
        """

        data = pd.DataFrame({'electrical_cost': np.array(electrical_cost_history).clip(min=0)})
        return data['electrical_cost'].sum()
