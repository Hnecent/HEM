from typing import List, Tuple, Union
import numpy as np


class EpisodeTracker:
    """Class for keeping track of current episode time steps for reading observations from data files.

    An EpisodeTracker object is shared amongst the environment, buildings in environment and all descendant 
    building devices. The object however, should be updated at the environment level only so that its changes 
    propagate to all other evironment decscendants.

    `simulation_start_time_step` and `simulation_end_time_step`
    are useful to separate training data from test data in the same data file. There may be one or more episodes betweeen
    `simulation_start_time_step` and `simulation_end_time_step` and their values should be defined in `schema` 
    or parsed to :py:class:`citylearn.citylearn.CityLearnEnv.__init__`. Both `simulation_start_time_step` and 
    `simulation_end_time_step` are used to select time series for building device and storage sizing as well as 
    action and observation space estimation in :py:class:`citylearn.buiLding.Building`.

    Parameters
    ----------
    simulation_start_time_step: int
        Time step to start reading from data files. 
    simulation_end_time_step: int
        Time step to end reading from data files.
    """

    def __init__(self, simulation_start_time_step: int, simulation_end_time_step: int):
        self.__episode = None
        self.__episode_start_time_step = None
        self.__episode_end_time_step = None
        self.__simulation_start_time_step = simulation_start_time_step
        self.__simulation_end_time_step = simulation_end_time_step
        self.reset_episode_index()

    @property
    def episode(self):
        """Current episode index"""

        return self.__episode

    @property
    def episode_time_steps(self):
        """Number of time steps in current episode split."""

        return (self.episode_end_time_step - self.episode_start_time_step) + 1

    @property
    def simulation_time_steps(self):
        """Number of time steps between `simulation_start_time_step` and `simulation_end_time_step`."""

        return (self.__simulation_end_time_step - self.__simulation_start_time_step) + 1

    @property
    def simulation_start_time_step(self):
        """Time step to start reading from data files."""

        return self.__simulation_start_time_step

    @property
    def simulation_end_time_step(self):
        """Time step to end reading from data files."""

        return self.__simulation_end_time_step

    @property
    def episode_start_time_step(self):
        """Start time step in current episode split."""

        return self.__episode_start_time_step

    @property
    def episode_end_time_step(self):
        """End time step in current episode split."""

        return self.__episode_end_time_step

    def next_episode(self, episode_time_steps: Union[int, List[Tuple[int, int]]], rolling_episode_split: bool,
                     random_episode_split: bool, random_seed: int):
        """Advance to next episode and set `episode_start_time_step` and `episode_end_time_step` for reading data files.
        
        Parameters
        ----------
        episode_time_steps: Union[int, List[Tuple[int, int]]], optional
            If type is `int`, it is the number of time steps in an episode. If type is `List[Tuple[int, int]]]` is provided, it is a list of 
            episode start and end time steps between `simulation_start_time_step` and `simulation_end_time_step`. Defaults to (`simulation_end_time_step` 
        - `simulation_start_time_step`) + 1. Will ignore `rolling_episode_split` if `episode_splits` is of type `List[Tuple[int, int]]]`.
        rolling_episode_split: bool, default: False
            True if episode sequences are split such that each time step is a candidate for `episode_start_time_step` otherwise, False to split episodes 
            in steps of `episode_time_steps`.
        random_episode_split: bool, default: False
            True if episode splits are to be selected at random during training otherwise, False to select sequentially.
        random_seed: int
            Random seed for selecting random episode splits
        """

        self.__episode += 1
        self.__next_episode_time_steps(
            episode_time_steps,
            rolling_episode_split,
            random_episode_split,
            random_seed,
        )

    def __next_episode_time_steps(self, episode_time_steps: Union[int, List[Tuple[int, int]]],
                                  rolling_episode_split: bool, random_episode_split: bool, random_seed: int):
        """Sets `episode_start_time_step` and `episode_end_time_step` for reading data files."""

        splits = None

        if isinstance(episode_time_steps, List):
            splits = episode_time_steps

        else:
            earliest_start_time_step = self.__simulation_start_time_step
            latest_start_time_step = (self.__simulation_end_time_step + 1) - episode_time_steps

            if rolling_episode_split:  # 是否滚动分割，即每个时间步都是一个episode
                start_time_steps = range(earliest_start_time_step, latest_start_time_step + 1)
            else:
                start_time_steps = range(earliest_start_time_step, latest_start_time_step + 1, episode_time_steps)

            end_time_steps = np.array(start_time_steps, dtype=int) + episode_time_steps - 1
            splits = np.array([start_time_steps, end_time_steps], dtype=int).T
            splits = splits.tolist()

        if random_episode_split:
            seed = int(random_seed * (self.episode + 1))
            nprs = np.random.RandomState(seed)
            ix = nprs.choice(len(splits) - 1)

        else:
            ix = self.episode % len(splits)

        self.__episode_start_time_step, self.__episode_end_time_step = splits[ix]

    def reset_episode_index(self):
        """Resets episode index to -1 before any simulation."""

        self.__episode = -1
