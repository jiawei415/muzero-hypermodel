import os
import copy
import glog
import time
import numpy
import torch
import pickle
import argparse
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
    def __init__(self, game_name, config=None):
        glog.info(f"this is game: {game_name}")
        # Load the game and the config from the module with the game name
        game_module = importlib.import_module("games." + game_name)
        Game = game_module.Game
        self.config = game_module.MuZeroConfig()
        self.init_config(game_name, config)
        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        self.game = Game(self.config)
        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint = {
            "action_mapping": None,
            "weights": None,
            "optimizer_state": None,
            "init_norm": None,
            "target_norm": None,
            "prior_model": None,
            "train_total_reward": 0,
            "train_episode_length": 0,
            "train_mean_value": 0,
            "test_total_reward": 0,
            "test_episode_length": 0,
            "test_mean_value": 0,
            "played_games": 0,
            "played_steps": 0,
            "training_steps": 0,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "lr": 0,
            "terminate": False,
        }
        actor = Actor(self.config)
        self.model, self.target_model, weights, summary = actor.initial_model()
        self.optimizer = actor.initial_optimizer(self.model)
        self.checkpoint["weights"] = copy.deepcopy(weights)
        self.summary = copy.deepcopy(summary)
        self.best_reward = float("-inf")
        # Workers
        self.self_play_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None
        self.test_worker = None
        self.debug_worker = None
        self.record_worker = None

    def init_config(self, game_name, config):
        for k, v in config.items():
            if k not in self.config.__dict__.keys():
                print(f'unrecognized config k: {k}, v: {v}, ignored')
                continue
            self.config.__dict__[k] = v
        if self.config.use_priormodel: self.config.priormodel = copy.deepcopy(self.config.hypermodel)
        if self.config.use_normalization: self.config.normalization = copy.deepcopy(self.config.hypermodel)
        if self.config.use_target_noise: self.config.target_noise = copy.deepcopy(self.config.hypermodel)
        if self.config.use_value_target_noise: self.config.target_noise[0] = 1
        if game_name == "deepsea": self.config.observation_shape = (1, 1, self.config.size**2)
        log_path = f"results/{game_name}_{self.config.seed}"
        self.config.game_filename = game_name
        self.config.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"{log_path}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}")

    def init_workers(self):
        # Initialize tensorboard
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
            "total_loss",
            "value_loss",
            "reward_loss",
            "policy_loss",
            "lr"
        ]
        self.palyer_logs_path = self.config.results_path + "/palyer_logs.csv"
        self.palyer_logs = pd.DataFrame(columns=self.keys)
        self.palyer_logs.to_csv(self.palyer_logs_path, sep="\t", index=False)

        debug_keys = [
            "sample_num",
            "mcts_value",
            "model_value",
            "target_model_value",
            "value_params.weight",
            "reward_params.weight",
            "state_params.weight"
        ]
        for i in self.config.action_space:
            debug_keys.extend([f"mcts_action_{i}", f"model_action_{i}"])
        self.debug_logs_path = self.config.results_path + "/debug_logs.csv"
        self.debug_logs = pd.DataFrame(columns=debug_keys)
        self.debug_logs.to_csv(self.debug_logs_path, sep="\t", index=False)

        # Initialize workers
        self.shared_storage_worker = shared_storage.SharedStorage(self.checkpoint, self.config)
        self.shared_storage_worker.set_info("terminate", False)
        if self.config.game_filename == "deepsea":
            self.shared_storage_worker.set_info('action_mapping', self.game.env._action_mapping)

        self.reanalyse_worker = replay_buffer.Reanalyse(self.target_model, self.config)
        self.replay_buffer_worker = replay_buffer.ReplayBuffer(self.reanalyse_worker, self.config)
        self.training_worker = trainer.Trainer(self.model, self.target_model, self.optimizer, self.config, self.writer)
        self.self_play_worker = self_play.SelfPlay(self.model, self.game, self.config)
        self.test_worker = self_play.TestPlay(self.model, self.game, self.config)
        self.debug_worker = debug.Debug(self.model, self.target_model, self.config, self.writer)
        if self.config.record_video:
            self.record_worker = self_play.RecordPlay(self.model, self.game, self.config)

    def train(self):
        self.init_workers()
        self.start_train = False
        played_games, played_steps, training_steps = 0, 0, 0
        for episode in range(self.config.total_episode):
            done = False
            game_history = self.self_play_worker.start_game()
            while not done and len(game_history.action_history) <= self.config.max_moves:
                done = self.self_play_worker.play_game(
                    game_history,
                    self.config.visit_softmax_temperature_fn(training_steps),
                    self.config.temperature_threshold,
                )
                played_steps += 1
                self.shared_storage_worker.set_info({"played_steps": played_steps})
                if played_games >= self.config.start_train and played_steps % self.config.train_frequency == 0:
                    self.start_train = True
                    train_times = self.config.train_per_paly(played_steps)
                    # for _ in tqdm(range(train_times)):
                    for _ in range(train_times):
                        if training_steps % self.config.checkpoint_interval == 0:
                            self.save_checkpoint()
                        index_batch, batch = self.replay_buffer_worker.get_batch()
                        priorities, losses = self.training_worker.train_game(batch, training_steps, played_steps)
                        if self.config.PER:
                            self.replay_buffer_worker.update_priorities(priorities, index_batch)
                        training_steps += 1
                        self.shared_storage_worker.set_info({"training_steps": training_steps})
            played_games += 1
            self.shared_storage_worker.set_info({"played_games": played_games})
            self.self_play_worker.close_game()
            self.replay_buffer_worker.save_game(game_history)
            self.shared_storage_worker.set_info(
                {
                    "train_total_reward": sum(game_history.reward_history),
                    "train_mean_value": numpy.mean(game_history.root_values),
                    "train_episode_length": len(game_history.action_history) - 1,
                }
            )
            if self.start_train:
                self.shared_storage_worker.set_info(losses)
                self.shared_storage_worker.set_info({"lr": self.optimizer.param_groups[0]["lr"]})
            self.test()
            self.debug()
            self.run_log(episode)
            if episode % (self.config.total_episode / 10) == 0:
                self.save_checkpoint(path=f"model{'%03d' % episode}.checkpoint")
            if self.config.record_video:
                self.record_worker.start_record()

        self.terminate_workers()

    def test(self):
        total_reward, mean_value, episode_length = 0, 0, 0
        # for i in tqdm(range(self.config.test_times)):
        for i in range(self.config.test_times):
            done = False
            game_history = self.test_worker.start_game()
            while not done and len(game_history.action_history) <= self.config.max_moves:
                done = self.test_worker.play_game(game_history)
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
        if total_reward/self.config.test_times > self.best_reward:
            self.best_reward = total_reward/self.config.test_times
            self.save_checkpoint(path="model_best.checkpoint")

    def debug(self):
        counter = self.shared_storage_worker.get_info("played_steps")
        init_state_value, actions_probability, hypermodel_std = self.debug_worker.start_debug()
        debug_log = [counter]
        for k, v in init_state_value.items():
            self.writer.add_histogram(f"5.Debug/value/{k}", numpy.array(v), counter)
            self.writer.add_scalar(f"5.Debug/value/{k}_mean", numpy.mean(numpy.array(v)), counter)
            debug_log.append(v)
        for k, v  in hypermodel_std.items():
            self.writer.add_scalar(f"5.Debug/params/{k}", v, counter)
            debug_log.append(v)
        for k, v in actions_probability.items():
            self.writer.add_histogram(f"5.Debug/action/{k}", numpy.array(v), counter)
            self.writer.add_scalar(f"5.Debug/action/{k}_mean", numpy.mean(numpy.array(v)), counter)
            debug_log.append(v)
        self.debug_logs.loc[counter] = debug_log
        self.debug_logs.to_csv(self.debug_logs_path, sep="\t", index=False)

    def run_log(self, counter):
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
            info["total_loss"],
            info["value_loss"],
            info["reward_loss"],
            info["policy_loss"],
            info["lr"],
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
            info["training_steps"] / max(1, info["played_steps"]), counter,
        )

        if self.start_train:
            self.writer.add_scalar("4.Trainer/1.Total_loss", info["total_loss"], counter)
            self.writer.add_scalar("4.Trainer/2.Value_loss", info["value_loss"], counter)
            self.writer.add_scalar("4.Trainer/3.Reward_loss", info["reward_loss"], counter)
            self.writer.add_scalar("4.Trainer/4.Policy_loss", info["policy_loss"], counter)
            self.writer.add_scalar("4.Trainer/5.Learning_rate", info["lr"], counter)

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
            self.save_checkpoint()
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
        self.record_worker = None

    def save_checkpoint(self, path=None):
        self.shared_storage_worker.set_info(
            {
                "weights": copy.deepcopy(self.model.get_weights()),
                "optimizer_state": copy.deepcopy(
                    models.dict_to_cpu(self.optimizer.state_dict())
                ),
                "init_norm": copy.deepcopy(self.model.init_norm),
                "target_norm": copy.deepcopy(self.model.target_norm),
                "prior_model": copy.deepcopy(self.model.prior_model),
            }
        )
        if self.config.save_model:
            self.shared_storage_worker.save_checkpoint(path=path)

    def evaluate(self, ckpt_path, render=False):
        self.checkpoint = torch.load(ckpt_path)
        if self.config.game_filename == "deepsea":
            self.game.env._action_mapping = self.checkpoint['action_mapping']
        self.model.set_weights(self.checkpoint["weights"])
        self.model.init_norm = self.checkpoint["init_norm"]
        self.model.target_norm = self.checkpoint["target_norm"]
        self.model.prior_model = self.checkpoint["prior_model"]
        self.record_worker = self_play.RecordPlay(self.model, self.game, self.config)
        total_reward = self.record_worker.start_record(render=render)
        print(f"total reward: {total_reward}")

class Actor:
    def __init__(self, config):
        self.config = config

    def initial_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = models.MuZeroNetwork(self.config).to(device)
        target_model = models.MuZeroNetwork(self.config).to(device)
        weigths = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        print(f"{str(model)}")
        target_model.set_weights(weigths)
        target_model.init_norm = model.init_norm
        target_model.target_norm = model.target_norm
        target_model.prior_model = model.prior_model
        if "cuda" not in str(next(model.parameters()).device):
            print("You are not training on GPU.\n")
        return model, target_model, weigths, summary

    def initial_optimizer(self, model):
        # Initialize the optimizer
        if self.config.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                [
                    {'params': (p for name, p in model.named_parameters() if 'hyper' not in name)},
                    {'params': (p for name, p in model.named_parameters() if 'hyper' in name), 'weight_decay': self.config.hyper_weight_decay}
                ],
                lr=self.config.lr_init,
                momentum=self.config.momentum,
                weight_decay=self.config.base_weight_decay,
            )
        elif self.config.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                [
                    {'params': (p for name, p in model.named_parameters() if 'hyper' not in name)},
                    {'params': (p for name, p in model.named_parameters() if 'hyper' in name), 'weight_decay': self.config.hyper_weight_decay}
                ],
                lr=self.config.lr_init,
                weight_decay=self.config.base_weight_decay,
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default="deepsea",
                        help='game name')
    parser.add_argument('--config', type=str, default="{}",
                        help="game config eg., {'seed':0,'total_episode':600,'train_frequency':50,'train_proportion':3,'hypermodel':[0,0,1],'use_priormodel':True,'td_steps':5,'value_loss_weight':0.25,'num_unroll_steps':10,'support_size':10}")
    parser.add_argument('--ckpt-path', type=str, default="",
                        help="checkpoint path for evaluation")
    parser.add_argument('--render', default=False, action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    args = get_args()
    muzero = MuZero(args.game, eval(args.config))
    if args.ckpt_path:
        muzero.evaluate(args.ckpt_path, args.render)
    else:
        muzero.train()

