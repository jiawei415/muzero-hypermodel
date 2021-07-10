import os
import sys
import copy
import glog
import time
import numpy
import torch
import pickle
import warnings
import importlib
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import models
import trainer
import debug
import self_play
import replay_buffer
import shared_storage

class MuZero:
    def __init__(self, game_name, config=None, split_resources_in=1):
        # Load the game and the config from the module with the game name
        game_module = importlib.import_module("games." + game_name)
        self.config = game_module.MuZeroConfig()
        self.config.game_filename = game_name
        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint = {
            "weights": None,
            "optimizer_state": None,
            "train_total_reward": 0,
            "train_episode_length": 0,
            "train_mean_value": 0,
            "test_total_reward": 0,
            "test_episode_length": 0,
            "test_mean_value": 0,
            "played_games": 0,
            "played_steps": 0,
            "training_steps": 0,
            "terminate": False,
        }
        # self.replay_buffer = {}
        cpu_actor = CPUActor(self.config)
        weights, summary, self.model, self.target_model, self.optimizer = cpu_actor.initial_model_and_optimizer()
        self.checkpoint["weights"] = copy.deepcopy(weights)
        self.summary = copy.deepcopy(summary)
        # Workers
        self.self_play_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None
        self.test_worker = None
        self.debug_worker = None

    def init_workers(self, log_in_tensorboard=True):
        if log_in_tensorboard or self.config.save_model:
            os.makedirs(self.config.results_path, exist_ok=True)
            config_logs_path = self.config.results_path + "/config_logs.csv"
            hp_table = []
            config_logs = pd.DataFrame(columns=["key", "value"])
            for i, (key, value) in enumerate(self.config.__dict__.items()):
                hp_table.extend([f"| {key} | {value} |"])
                config_logs.loc[i] = [key, value]
            config_logs.to_csv(config_logs_path, sep="\t", index=False)
            print("\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n")
            self.writer = SummaryWriter(self.config.results_path)
            self.writer.add_text("Hyperparameters", "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),)
            self.writer.add_text("Model summary", self.summary,)
            self.keys = [
                "train_total_reward",
                "train_mean_value",
                "train_episode_length",
                "test_total_reward",
                "test_mean_value",
                "test_episode_length",
                "played_games",
                "played_steps",
                "training_steps",
            ]
            self.palyer_logs_path = self.config.results_path + "/palyer_logs.csv"
            self.palyer_logs = pd.DataFrame(columns=self.keys)
            self.palyer_logs.to_csv(self.palyer_logs_path, sep="\t", index=False)

        # Initialize workers
        self.shared_storage_worker = shared_storage.SharedStorage(self.checkpoint, self.config)
        self.shared_storage_worker.set_info("terminate", False)
        
        self.reanalyse_worker = replay_buffer.Reanalyse(self.target_model, self.config)
        self.replay_buffer_worker = replay_buffer.ReplayBuffer(self.reanalyse_worker, self.config)
        self.training_worker = trainer.Trainer(self.model, self.target_model, self.optimizer, self.config, self.writer)
        self.self_play_worker = self_play.SelfPlay(self.model, self.config)
        self.test_worker = self_play.TestPlay(self.model, self.config)
        self.debug_worker = debug.Debug(self.model, self.config, self.writer)

    def train(self, log_in_tensorboard=True):
        self.init_workers(log_in_tensorboard=log_in_tensorboard)
        played_games = 0
        played_steps = 0
        training_steps = 0
        for episode in range(self.config.episode):
            done = False
            game_history = self.self_play_worker.start_game()
            while not done and len(game_history.action_history) <= self.config.max_moves:
                done = self.self_play_worker.play_game(
                    game_history,
                    self.config.visit_softmax_temperature_fn(
                        trained_steps=self.shared_storage_worker.get_info("training_steps")
                    ),
                    self.config.temperature_threshold,
                )
                played_steps += 1
                self.shared_storage_worker.set_info({"played_steps": played_steps})
                if played_games >= self.config.start_train and played_steps % self.config.train_frequency == 0:       
                    train_times = self.config.train_per_paly(played_steps)
                    # for _ in tqdm(range(train_times)):
                    for _ in range(train_times):
                        if training_steps % self.config.checkpoint_interval == 0:
                            self.debug()
                            self.shared_storage_worker.set_info(
                                {
                                    "weights": copy.deepcopy(self.model.get_weights()),
                                    "optimizer_state": copy.deepcopy(
                                        models.dict_to_cpu(self.optimizer.state_dict())
                                    ),
                                }
                            )
                            if self.config.save_model: self.shared_storage_worker.save_checkpoint()
                        index_batch, batch = self.replay_buffer_worker.get_batch()
                        priorities = self.training_worker.train_game(batch, training_steps)
                        if self.config.PER:
                            self.replay_buffer_worker.update_priorities(priorities, index_batch)
                        training_steps += 1
                        self.shared_storage_worker.set_info({"training_steps": training_steps})                        
            played_games += 1
            self.shared_storage_worker.set_info({"played_games": played_games,})         
            self.self_play_worker.close_game()
            self.replay_buffer_worker.save_game(game_history)
            self.shared_storage_worker.set_info(
                {
                    "train_total_reward": sum(game_history.reward_history),
                    "train_mean_value": numpy.mean(game_history.root_values),
                    "train_episode_length": len(game_history.action_history) - 1,
                }
            )
            self.test()
            self.player_log(episode)
    
        self.terminate_workers()

    def test(self, model_path=None):
        if model_path is not None:
            self.model.load_state_dict(model_path)
        total_reward, mean_value, episode_length = 0, 0, 0
        # for i in tqdm(range(self.config.test_times)):
        for i in range(self.config.test_times):
            done = False
            game_history = self.test_worker.start_game()
            while not done and len(game_history.action_history) <= self.config.max_moves:
                done = self.test_worker.play_game(game_history, render=False)
            self.test_worker.close_game()
            total_reward += sum(game_history.reward_history)
            mean_value += numpy.mean(game_history.root_values)
            episode_length += len(game_history.action_history) - 1      
        self.shared_storage_worker.set_info(
            {
                "test_total_reward": total_reward/self.config.test_times,
                "test_mean_value": mean_value/self.config.test_times,
                "test_episode_length": episode_length/self.config.test_times,
            }
        )
        
    def debug(self):
        counter = self.shared_storage_worker.get_info("training_steps")
        self.debug_worker.start_debug(counter)

    def player_log(self, counter):
        info = self.shared_storage_worker.get_info(self.keys)
        palyer_log = [
            info["train_total_reward"],
            info["train_mean_value"],
            info["train_episode_length"],
            info["test_total_reward"],
            info["test_mean_value"],
            info["test_episode_length"],
            info["played_games"],
            info["played_steps"],
            info["training_steps"],
        ]
        self.palyer_logs.loc[counter] = palyer_log
        self.palyer_logs.to_csv(self.palyer_logs_path, sep="\t", index=False)

        self.writer.add_scalar("1.TrainPlayer/1.Total_reward", info["train_total_reward"], counter)
        self.writer.add_scalar("1.TrainPlayer/2.Mean_value", info["train_mean_value"], counter)
        self.writer.add_scalar("1.TrainPlayer/3.Episode_length", info["train_episode_length"], counter)

        self.writer.add_scalar("2.TestPlayer/1.Total_reward", info["test_total_reward"], counter)
        self.writer.add_scalar("2.TestPlayer/2.Mean_value", info["test_mean_value"], counter)
        self.writer.add_scalar("2.TestPlayer/3.Episode_length", info["test_episode_length"], counter)

        self.writer.add_scalar("3.Workers/1.Played_games", info["played_games"], counter)
        self.writer.add_scalar("3.Workers/2.Played_steps", info["played_steps"], counter)
        self.writer.add_scalar("3.Workers/3.Training_steps", info["training_steps"], counter)
        self.writer.add_scalar("3.Workers/4.Training_steps_per_played_step_ratio",
            info["training_steps"] / max(1, info["played_steps"]),
            counter,
        )
        print(
            f'Test reward: {info["test_total_reward"]}. ' +
            f'Train reward: {info["train_total_reward"]}. ' +
            f'Training steps: {info["training_steps"]}. '+
            f'Played games: {info["played_games"]}. ' +
            f'Played steps: {info["played_steps"]}.',
        )

    def terminate_workers(self):
        """
        Softly terminate the running tasks and garbage collect the workers.
        """
        if self.config.save_model:
            # Persist replay buffer to disk
            print("\n\nPersisting replay buffer games to disk...")
            saved_keys = ['played_games', 'played_steps']
            saved_info = self.shared_storage_worker.get_info(saved_keys)
            pickle.dump(
                {
                    "buffer": self.replay_buffer_worker.buffer,
                    "played_games": saved_info["played_games"],
                    "played_steps": saved_info["played_steps"],
                },
                open(os.path.join(self.config.results_path, "replay_buffer.pkl"), "wb"),
            )
        if self.shared_storage_worker:
            self.shared_storage_worker.set_info("terminate", True)
            self.checkpoint = self.shared_storage_worker.get_checkpoint()

        if self.replay_buffer_worker:
            self.replay_buffer = self.replay_buffer_worker.get_buffer()

        print("\nShutting down workers...")

        self.self_play_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None
        self.test_worker = None
        self.debug_worker = None


class CPUActor:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self, config):
        self.config = config

    def initial_model_and_optimizer(self):
        model = models.MuZeroNetwork(self.config)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        target_model = models.MuZeroNetwork(self.config)
        target_model.load_state_dict(model.state_dict())
        value_normal, reward_normal, state_normal = self.config.normalization
        if value_normal:
            target_model.init_value_norm = model.init_value_norm
            target_model.target_value_norm = model.target_value_norm
            target_model.value_prior_params = model.value_prior_params
        if reward_normal:
            target_model.init_reward_norm = model.init_reward_norm
            target_model.target_reward_norm = model.target_reward_norm
            target_model.reward_prior_params = model.reward_prior_params
        if state_normal:
            target_model.init_state_norm = model.init_state_norm
            target_model.target_state_norm = model.target_state_norm
            target_model.state_prior_params = model.state_prior_params
        # print("\n", model)
        weigths = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        optimizer = self.initial_optimizer(model)
        if "cuda" not in str(next(model.parameters()).device):
            print("You are not training on GPU.\n")
        return weigths, summary, model, target_model, optimizer

    def initial_optimizer(self, model):
        # Initialize the optimizer
        if self.config.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.config.lr_init,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.lr_init,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"{self.config.optimizer} is not implemented. You can change the optimizer manually in trainer.py."
            )
        # if initial_checkpoint["optimizer_state"] is not None:
        #     print("Loading optimizer...\n")
        #     self.optimizer.load_state_dict(
        #         copy.deepcopy(initial_checkpoint["optimizer_state"])
        #     )
        return optimizer


if __name__ == "__main__": 
    warnings.filterwarnings('ignore')
    try:
        game = sys.argv[1]
    except:
        game = "cartpole"
    glog.info(f"this is game: {game}")
    muzero = MuZero(game)
    muzero.train()
    