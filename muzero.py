import copy
import glog
import importlib
import math
import os
import pickle
import sys
import time
from tqdm import tqdm

import numpy
# import ray
import torch
from torch.utils.tensorboard import SummaryWriter


import models
import replay_buffer
import self_play
import shared_storage
import trainer
import warnings
warnings.filterwarnings('ignore')

class MuZero:
    """
    Main class to manage MuZero.

    Args:
        game_name (str): Name of the game module, it should match the name of a .py file
        in the "./games" directory.

        config (dict, MuZeroConfig, optional): Override the default config of the game.

        split_resources_in (int, optional): Split the GPU usage when using concurent muzero instances.

    Example:
        >>> muzero = MuZero("cartpole")
        >>> muzero.train()
        >>> muzero.test(render=True)
    """

    def __init__(self, game_name, config=None, split_resources_in=1):
        # Load the game and the config from the module with the game name
        try:
            game_module = importlib.import_module("games." + game_name)
            self.Game = game_module.Game
            self.config = game_module.MuZeroConfig()
            self.config.game_filename = game_name
        except ModuleNotFoundError as err:
            print(
                f'{game_name} is not a supported game name, try "cartpole" or refer to the documentation for adding a new game.'
            )
            raise err

        # Overwrite the config
        if config:
            if type(config) is dict:
                for param, value in config.items():
                    setattr(self.config, param, value)
            else:
                self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint = {
            "weights": None,
            "optimizer_state": None,
            "total_reward": 0,
            "muzero_reward": 0,
            "opponent_reward": 0,
            "episode_length": 0,
            "mean_value": 0,
            "training_step": 0,
            "lr": 0,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "num_played_games": 0,
            "num_played_steps": 0,
            "num_reanalysed_games": 0,
            "terminate": False,
        }
        self.replay_buffer = {}

        cpu_actor = CPUActor()
        weights, summary, self.model, self.target_model = cpu_actor.get_initial_weights(self.config)
        self.checkpoint["weights"] = copy.deepcopy(weights)
        self.summary = copy.deepcopy(summary)
        # Workers
        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def init_workers(self, log_in_tensorboard=True):
        """
        Spawn ray workers and launch the training.

        Args:
            log_in_tensorboard (bool): Start a testing worker and log its performance in TensorBoard.
        """
        if log_in_tensorboard or self.config.save_model:
            os.makedirs(self.config.results_path, exist_ok=True)
            # self.test_worker = self_play.SelfPlay(self.checkpoint, self.Game, self.config, self.config.seed)
            self.writer = SummaryWriter(self.config.results_path)
            print("\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n")
            hp_table = [f"| {key} | {value} |" for key, value in self.config.__dict__.items()]
            self.writer.add_text("Hyperparameters", "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),)
            self.writer.add_text("Model summary", self.summary,)
            self.keys = [
                "total_reward",
                "muzero_reward",
                "opponent_reward",
                "episode_length",
                "mean_value",
                "training_step",
                "lr",
                "total_loss",
                "value_loss",
                "reward_loss",
                "policy_loss",
                "num_played_games",
                "num_played_steps",
                "num_reanalysed_games",
            ]

        # Initialize workers
        self.shared_storage_worker = shared_storage.SharedStorage(self.checkpoint, self.config)
        self.shared_storage_worker.set_info("terminate", False)

        self.training_worker = trainer.Trainer(self.model, self.checkpoint, self.config, self.writer)
        self.self_play_worker = self_play.SelfPlay(self.model, self.checkpoint, self.Game, self.config, self.writer)
        self.reanalyse_worker = replay_buffer.Reanalyse(self.target_model, self.checkpoint, self.shared_storage_worker, self.config)
        self.replay_buffer_worker = replay_buffer.ReplayBuffer(self.checkpoint, self.replay_buffer, self.reanalyse_worker, self.config)

    def train(self, log_in_tensorboard=True):
        self.init_workers(log_in_tensorboard=log_in_tensorboard)
        for counter in range(self.config.episode):
            self.self_play_worker.continuous_self_play(self.shared_storage_worker, self.replay_buffer_worker)
            num_played_games = self.shared_storage_worker.get_info('num_played_games')
            if num_played_games % 2 == 0:
                train_times = self.config.train_times(num_played_games)
                # for _ in tqdm(range(train_times)):
                for _ in range(train_times):
                    training_step = self.shared_storage_worker.get_info("training_step")
                    if training_step % self.config.target_update_freq == 0:
                        # print(f"update target model")
                        self.target_model.load_state_dict(self.model.state_dict())
                    self.training_worker.continuous_update_weights(self.replay_buffer_worker, self.shared_storage_worker)

            if log_in_tensorboard:
                # self.test_worker.continuous_self_play(self.shared_storage_worker, None, True)
                self.logging_loop(counter)


        if self.config.save_model:
            # Persist replay buffer to disk
            print("\n\nPersisting replay buffer games to disk...")
            saved_keys = ['num_played_games', 'num_played_steps', 'num_reanalysed_games']
            saved_info = self.shared_storage_worker.get_info(saved_keys)
            pickle.dump(
                {
                    "buffer": self.replay_buffer_worker.buffer,
                    "num_played_games": saved_info["num_played_games"],
                    "num_played_steps": saved_info["num_played_steps"],
                    "num_reanalysed_games": saved_info["num_reanalysed_games"],
                },
                open(os.path.join(self.config.results_path, "replay_buffer.pkl"), "wb"),
            )

        self.terminate_workers()

    def logging_loop(self, counter):
        """
        Keep track of the training performance.
        """
        # Updating the training performance
        info = self.shared_storage_worker.get_info(self.keys)
        try:
            self.writer.add_scalar(
                "1.Total_reward/1.Total_reward", info["total_reward"], counter,
            )
            self.writer.add_scalar(
                "1.Total_reward/2.Mean_value", info["mean_value"], counter,
            )
            self.writer.add_scalar(
                "1.Total_reward/3.Episode_length", info["episode_length"], counter,
            )
            self.writer.add_scalar(
                "1.Total_reward/4.MuZero_reward", info["muzero_reward"], counter,
            )
            self.writer.add_scalar(
                "1.Total_reward/5.Opponent_reward",
                info["opponent_reward"],
                counter,
            )
            self.writer.add_scalar(
                "2.Workers/1.Self_played_games", info["num_played_games"], counter,
            )
            self.writer.add_scalar(
                "2.Workers/2.Training_steps", info["training_step"], counter
            )
            self.writer.add_scalar(
                "2.Workers/3.Self_played_steps", info["num_played_steps"], counter
            )
            self.writer.add_scalar(
                "2.Workers/4.Reanalysed_games",
                info["num_reanalysed_games"],
                counter,
            )
            self.writer.add_scalar(
                "2.Workers/5.Training_steps_per_self_played_step_ratio",
                info["training_step"] / max(1, info["num_played_steps"]),
                counter,
            )
            self.writer.add_scalar("2.Workers/6.Learning_rate", info["lr"], counter)
            self.writer.add_scalar(
                "3.Loss/1.Total_weighted_loss", info["total_loss"], counter
            )
            self.writer.add_scalar("3.Loss/Value_loss", info["value_loss"], counter)
            self.writer.add_scalar("3.Loss/Reward_loss", info["reward_loss"], counter)
            self.writer.add_scalar("3.Loss/Policy_loss", info["policy_loss"], counter)
            print(
                f'Counter: {counter}/{self.config.episode}. Last play reward: {info["total_reward"]:.2f}. Training step: {info["training_step"]}. Played step: {info["num_played_steps"]}. Played games: {info["num_played_games"]}',
                # end="\r",
            )
        except KeyboardInterrupt:
            pass

    def terminate_workers(self):
        """
        Softly terminate the running tasks and garbage collect the workers.
        """
        if self.shared_storage_worker:
            self.shared_storage_worker.set_info("terminate", True)
            self.checkpoint = self.shared_storage_worker.get_checkpoint()

        if self.replay_buffer_worker:
            self.replay_buffer = self.replay_buffer_worker.get_buffer()

        print("\nShutting down workers...")

        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None


class CPUActor:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self):
        pass

    def get_initial_weights(self, config):
        model = models.MuZeroNetwork(config)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        target_model = models.MuZeroNetwork(config)
        target_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        target_model.load_state_dict(model.state_dict())
        value_normal, reward_normal, state_normal = config.normalization
        if value_normal:
            target_model.init_value_norm = model.init_value_norm
            target_model.target_value_norm = model.target_value_norm
        if reward_normal:
            target_model.init_reward_norm = model.init_reward_norm
            target_model.target_reward_norm = model.target_reward_norm
        if state_normal:
            target_model.init_state_norm = model.init_state_norm
            target_model.target_state_norm = model.target_state_norm
        # print("\n", model)
        weigths = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        return weigths, summary, model, target_model


if __name__ == "__main__":
    try:
        game = sys.argv[1]
    except:
        game = "cartpole"
    glog.info(f"this is game: {game}")
    muzero = MuZero(game)
    muzero.train()
    