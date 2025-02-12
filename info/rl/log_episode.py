import sys
import numpy as np
from tqdm import tqdm

def log_episode(cumulative_reward:float, episode_length:int, step:tuple[int, int], window_size:int=100) -> None:
    '''
    Records Reward data in an RL training session.
    Args:
            `reward`(float): Cumulated reward at the end of an episode.
            `length`(int): The length of the episode computed in steps.
            `step`(tuple[int, int]): The current (global step out of max_steps, max_steps).  
            `window_size`(int): Other statistics are computed using a window of last elements.
    '''
    _RLTrainLogger.log_episode(cumulative_reward=cumulative_reward, 
                                    episode_length=episode_length, 
                                    step=step,
                                    window_size=window_size)


class _RLTrainLogger():
    _instance: '_RLTrainLogger' = None

    def __new__(cls, total_steps:int= 1e5, window_size=100):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, total_steps:int= 1e5, window_size=100):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.window_size:int = window_size
        self.reward_window = []
        self.episodes_history = []
        self.max_recorded_reward:float = -float('inf')
        self.max_steps = total_steps
        self.display = tqdm(total = total_steps, leave=False)

    def _update(self, step:int):
        window_mean = np.mean(self.reward_window)
        window_std = np.std(self.reward_window)
        self.display.set_description(f"[Episode {len(self.episodes_history)}]")
        self.display.set_postfix(
            {
                "r_max":self.max_recorded_reward,
                "r_avg":window_mean,
                "r_std": window_std,
            }
        )
        self.display.n = min(step, self.max_steps)
        self.display.refresh()


    @staticmethod
    def log_episode(cumulative_reward:float, episode_length:int, step:tuple[int, int], window_size:int=100) -> None:
        '''
        Args:
            `reward`(float): Cumulated reward at the end of an episode.
            `length`(int): The length of the episode computed in steps.
            `step`(tuple[int, int]): The current (global step, max_steps).  
            `window_size`(int): Other statistics are computed using a window of last elements.
        '''
        assert isinstance(step, tuple), "Note `step` must be a tuple[int, int]"
        assert episode_length > 0, "Episodes cannot last 0 steps or less."
        assert window_size >= 5, "Please use a high window size in order to get good info"
        
        logger = _RLTrainLogger(total_steps=step[1], window_size=window_size)

        if cumulative_reward > logger.max_recorded_reward:
           logger.max_recorded_reward = cumulative_reward


        logger.episodes_history.append((cumulative_reward, episode_length, step))
        logger.reward_window.append(cumulative_reward)

        if(len(logger.reward_window) > logger.window_size):
            logger.reward_window.pop(0)

        logger._update(step[0])