import numpy
import torch
from utils import MCTS, GameHistory, play_action

class SelfPlay:
    def __init__(self, model, game, config):
        self.config = config
        self.game = game
        self.use_value_noise, self.use_reward_noise, _ = config.target_noise
        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        self.model = model
        self.noise_dim = int(self.config.hyper_inp_dim)
             
    def start_game(self):
        observation = self.game.reset()
        assert (
            len(numpy.array(observation).shape) == 3
        ), f"Observation should be 3 dimensionnal instead of {len(numpy.array(observation).shape)} dimensionnal. Got observation of shape: {numpy.array(observation).shape}"
        assert (
            numpy.array(observation).shape == self.config.observation_shape
        ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {numpy.array(observation).shape}."
        noise_z = numpy.random.normal(0, 1, [1, self.noise_dim]) * self.config.normal_noise_std

        game_history = GameHistory()
        game_history.noise_history = noise_z
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
        game_history.to_play_history.append(self.game.to_play())
        if self.use_value_noise or self.use_reward_noise:
            game_history.unit_sphere_history.append(self.sample_unit_sphere())

        return game_history

    def play_game(self, game_history, temperature, temperature_threshold):    
        self.model.eval()
        with torch.no_grad():
            stacked_observations = game_history.get_stacked_observations(
                -1,
                self.config.stacked_observations,
            )
            # Choose the action
            root, mcts_info = MCTS(self.config).run(
                game_history.noise_history,
                self.config.num_simulations,
                self.model,
                stacked_observations,
                self.game.legal_actions(),
                self.game.to_play(),
                True,
            )
            # action = self.select_action(root, temperature if not temperature_threshold or len(game_history.action_history) < temperature_threshold else 0)
            action = play_action(root, temperature if not temperature_threshold or len(game_history.action_history) < temperature_threshold else 0, self.config)
            observation, reward, done = self.game.step(action)

            game_history.store_search_statistics(root, self.config.action_space)
            # Next batch
            game_history.action_history.append(action)
            game_history.observation_history.append(observation)
            game_history.reward_history.append(reward)
            game_history.to_play_history.append(self.game.to_play())
            if self.use_value_noise or self.use_reward_noise:
                game_history.unit_sphere_history.append(self.sample_unit_sphere())

            return done

    def close_game(self):
        self.game.close()

    def sample_unit_sphere(self):
        noise = numpy.random.normal(0, 1, [1, self.noise_dim]) * self.config.target_noise_std
        noise /= numpy.sqrt((noise**2).sum())
        return noise

    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = numpy.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[numpy.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = numpy.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = numpy.random.choice(actions, p=visit_count_distribution)

        return action


class TestPlay:
    def __init__(self, model, game, config):
        self.config = config
        self.game = game
        self.model = model
        self.noise_dim = int(self.config.hyper_inp_dim)
        self.fix_noise = torch.normal(0, 1, [1, self.noise_dim]) * self.config.normal_noise_std  
        
    def start_game(self, fix_noise=False):
        observation = self.game.reset()
        if fix_noise:
            noise_z = self.fix_noise
        else:
            noise_z = numpy.random.normal(0, 1, [1, self.noise_dim]) * self.config.normal_noise_std
        game_history = GameHistory()
        game_history.noise_history = noise_z
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
        game_history.to_play_history.append(self.game.to_play())

        return game_history

    def play_game(self, game_history):
        self.model.eval()
        with torch.no_grad():
            stacked_observations = game_history.get_stacked_observations(
                -1,
                self.config.stacked_observations,
            )
            # Choose the action
            root, mcts_info = MCTS(self.config).run(
                game_history.noise_history,
                self.config.num_simulations,
                self.model,
                stacked_observations,
                self.game.legal_actions(),
                self.game.to_play(),
                True,
            )
            # action = self.select_action(root, temperature=0)
            action = play_action(root, 0, self.config)
            observation, reward, done = self.game.step(action)
            game_history.store_search_statistics(root, self.config.action_space)
            # Next batch
            game_history.action_history.append(action)
            game_history.observation_history.append(observation)
            game_history.reward_history.append(reward)
            game_history.to_play_history.append(self.game.to_play())
            return done

    def close_game(self):
        self.game.close()

    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = numpy.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[numpy.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = numpy.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = numpy.random.choice(actions, p=visit_count_distribution)

        return action


class RecordPlay:
    def __init__(self, model, game, config):
        self.config = config
        self.game = game
        self.model = model
        self.noise_dim = int(self.config.hyper_inp_dim)

    def start_record(self, render=False):
        done = False
        noise_z = numpy.random.normal(0, 1, [1, self.noise_dim]) * self.config.normal_noise_std
        observation = self.game.reset()
        if render:
            self.game.render()
        game_history = GameHistory()
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        self.model.eval()
        while not done and len(game_history.action_history) <= self.config.max_moves:
            with torch.no_grad():
                stacked_observations = game_history.get_stacked_observations(
                    -1,
                    self.config.stacked_observations,
                )
                # Choose the action
                if self.config.use_mcts:
                    root, mcts_info = MCTS(self.config).run(
                        noise_z,
                        self.config.num_simulations,
                        self.model,
                        stacked_observations,
                        self.game.legal_actions(),
                        self.game.to_play(),
                        True,
                    )
                    # action = self.select_action(root, temperature=0)
                    action = play_action(root, 0, self.config)
                else:
                    obs = torch.tensor(stacked_observations).float().unsqueeze(0)
                    noise_z = torch.tensor(noise_z).float()
                    _, _, logits, _, _ = self.model.initial_inference(
                        obs.to(next(self.model.parameters()).device),
                        noise_z.to(next(self.model.parameters()).device)
                    )
                    policy = torch.softmax(logits, dim=1).squeeze().numpy()
                    action = numpy.random.choice(self.config.action_space, 1, p=policy).squeeze()
                observation, reward, done = self.game.step(action)
                if render:
                    self.game.render()
                game_history.observation_history.append(observation)
                game_history.action_history.append(action)
                game_history.reward_history.append(reward)
                game_history.to_play_history.append(self.game.to_play())
        return sum(game_history.reward_history)

    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = numpy.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[numpy.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = numpy.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = numpy.random.choice(actions, p=visit_count_distribution)

        return action
