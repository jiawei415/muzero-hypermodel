import os
import gym
import datetime
import numpy as np
from gym import spaces
from gym.utils import seeding
from .basicconfig import BasicConfig
from .abstract_game import AbstractGame


class MuZeroConfig(BasicConfig):
    def __init__(self):
        super(MuZeroConfig, self).__init__()
        ### Game
        self.size = 30
        self.use_reward_wrapper = True
        self.observation_shape = (1, 1, self.size**2)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(2))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length

        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3] + "_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))  # Path to store the model weights and TensorBoard logs

class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = DeepSeaEnv(MuZeroConfig().size)
        if seed is not None:
            self.env.seed(seed)

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """

        observation, reward, done, _ = self.env.step(action)
        obs = np.reshape(observation, (1, -1))
        return np.array([obs]), reward, done

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return list(range(2))

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        observation = self.env.reset()
        obs = np.reshape(observation, (1, -1))
        return np.array([obs])

    def close(self):
        """
        Properly close the game.
        """
        self.env.close()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = {
            0: "Go left",
            1: "Go right",
        }
        return f"{action_number}. {actions[action_number]}"

class DeepSeaEnv(gym.Env):
    """
    Deep sea example.

    For more information, see papers:
    [1] https://arxiv.org/abs/1703.07608
    [2] https://arxiv.org/abs/1806.03335
    """
    def __init__(self, size, deterministic=True, unscaled_move_cost=0.01):
        super().__init__()

        self._size = size
        self._deterministic = deterministic # config.get("deterministic", True)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self._size, self._size), dtype=np.int32)
        self.action_space = spaces.Discrete(n=2)

        self._row = 0
        self._column = 0
        self._unscaled_move_cost = unscaled_move_cost # config.get("move_cost", 0.01)
        self._action_mapping = np.ones([self._size, self._size])
        self.use_move_cost = True

        assert self._unscaled_move_cost * self._size <= 1, (
            "Please decrease the move cost. Otherwise the optimal decision is not go right."
        )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        self._row = 0
        self._column = 0
        return self._get_observation()

    def _get_observation(self):
        obs = np.zeros(shape=(self._size, self._size), dtype=np.float32)
        if self._row >= self._size:  # End of episode null observation
            return obs
        obs[self._row, self._column] = 1.
        return obs

    def step(self, action: int):
        reward = 0. if MuZeroConfig().use_reward_wrapper else -1.
        action_right = action == 1 # self._action_mapping[self._row, self._column]

        # Reward calculation
        if self._column == self._size - 1 and action_right:
            reward += 1.
        if not self._deterministic:  # Noisy rewards on the 'end' of chain.
            if self._row == self._size - 1 and self._column in [0, self._size - 1]:
                reward += np.random.rand()

        if action_right:
            if np.random.rand() > 1 / self._size or self._deterministic:
                self._column = np.clip(self._column + 1, 0, self._size - 1)
            if self.use_move_cost:
                reward -= self._unscaled_move_cost / self._size
        else:
            # You were on the right path and went wrong
            self._column = np.clip(self._column - 1, 0, self._size - 1)
        self._row += 1

        observation = self._get_observation()
        done = False
        info = {}
        if self._row == self._size:
            done = True

        return observation, reward, done, info

    def render(self, mode='human'):
        print(self._get_observation())
