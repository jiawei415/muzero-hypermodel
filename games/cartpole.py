import gym
import numpy
from gym.wrappers import TimeLimit, Monitor
from .basicconfig import BasicConfig
from .abstract_game import AbstractGame


class MuZeroConfig(BasicConfig):
    def __init__(self):
        super(MuZeroConfig, self).__init__()
        self.observation_shape = (1, 1, 4)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(2))  # Fixed list of all possible actions. You should only edit the length

class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, config):
        self.env = gym.make("CartPole-v1")
        if config.seed is not None:
            self.env.seed(config.seed)
            self.env = TimeLimit(self.env, max_episode_steps=500)
        if config.record_video:
            record_frequency = config.record_frequency
            video_callable = lambda episode: episode % record_frequency == 0
            self.env = TimeLimit(self.env, max_episode_steps=500)
            self.env = Monitor(self.env, f'{config.results_path}/videos', video_callable=video_callable, force=True)

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """

        observation, reward, done, _ = self.env.step(action)
        return numpy.array([[observation]]), reward, done

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
        return numpy.array([[observation]])

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
            0: "Push cart to the left",
            1: "Push cart to the right",
        }
        return f"{action_number}. {actions[action_number]}"
