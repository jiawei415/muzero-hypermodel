import numpy
import torch
import importlib
import pandas as pd
from tqdm import tqdm
from utils import MCTS, GameHistory, support_to_scalar

class Debug:
    def __init__(self, model, target_model, config, writer):
        self.config = config
        game_module = importlib.import_module("games." + self.config.game_filename)
        Game = game_module.Game
        self.game = Game(config)
        self.model = model
        self.target_model = target_model
        self.writer = writer
        self.noise_dim = int(self.config.hyper_inp_dim)
        self.actions_log = dict()
        self.value_log = {"mcts_value":[], "target_model_value":[], "model_value":[]}
        keys = ["counter", "mcts_value", "target_model_value", "model_value"]
        for i in self.game.legal_actions():
            keys.extend([f"mcts_action_{i}", f"model_action_{i}"])
            self.actions_log[f"mcts_action_{i}"] = []
            self.actions_log[f"model_action_{i}"] = []
        keys.extend(["value_params", "reward_params", "state_params"])
        keys.extend(["value_params.weight", "reward_params.weight", "state_params.weight"])
        self.debug_logs_path = self.config.results_path + "/debug_logs.csv"
        self.debug_logs = pd.DataFrame(columns=keys)
        self.debug_logs.to_csv(self.debug_logs_path, sep="\t", index=False)

    def start_debug(self, counter):
        observation = self.game.reset()
        game_history = GameHistory()
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        value_params, reward_params, state_params = [], [], []
        with torch.no_grad():
            stacked_observations = game_history.get_stacked_observations(
                -1,
                self.config.stacked_observations,
            )
            # for i in tqdm(range(self.config.debug_times)):
            for i in range(self.config.debug_times):
                noise_z = numpy.random.normal(0, 1, [1, self.noise_dim]) * self.config.normal_noise_std
                root, mcts_info = MCTS(self.config).run(
                    noise_z,
                    self.config.num_simulations,
                    self.model,
                    stacked_observations,
                    self.game.legal_actions(),
                    self.game.to_play(),
                    True,
                )
                self.value_log['mcts_value'].append(root.value())
                debug_observations = torch.tensor(stacked_observations).float().unsqueeze(0)
                noise_z = torch.tensor(noise_z).float()
                target_model_value, _, _, _, _ = self.target_model.initial_inference(
                        debug_observations.to(next(self.target_model.parameters()).device),
                        noise_z.to(next(self.target_model.parameters()).device)
                    )
                target_model_value = support_to_scalar(target_model_value, self.config.support_size).item()
                self.value_log['target_model_value'].append(target_model_value)
                model_value, _, debug_logits, _, _ = self.model.initial_inference(
                        debug_observations.to(next(self.model.parameters()).device),
                        noise_z.to(next(self.model.parameters()).device)
                    )
                model_value = support_to_scalar(model_value, self.config.support_size).item()
                self.value_log['model_value'].append(model_value)
                debug_policy = torch.softmax(debug_logits, dim=1).squeeze()
                for j in self.config.action_space:
                    self.actions_log[f"mcts_action_{j}"].append(root.children[j].prior)
                    self.actions_log[f"model_action_{j}"].append(debug_policy[j].item())
                debug_params = self.model.debug(noise_z.to(next(self.model.parameters()).device))
                for k, v in debug_params.items():
                    if "value" in k and v is not None:
                        value_params.append(v)
                    elif "reward" in k and v is not None:
                        reward_params.append(v)
                    elif "state" in k and v is not None:
                        state_params.append(v)
            value_params_std = self.calculation_std(value_params)
            reward_params_std = self.calculation_std(reward_params)
            state_params_std = self.calculation_std(state_params)
            params_std = {"value_params": value_params_std, "reward_params": reward_params_std, "state_params": state_params_std}
        hypermodel_std = self.model.get_hypermodel()
        self.debug_log(params_std, hypermodel_std, counter)
        self.game.close()

    def debug_log(self, debug_params, hypermodel_std, counter):
        debug_log = [counter]
        for k, v in self.value_log.items():
            self.writer.add_histogram(f"5.Debug/value/{k}", numpy.array(v), counter)
            self.writer.add_scalar(f"5.Debug/value/{k}_mean", numpy.mean(numpy.array(v)), counter)
            debug_log.append(v)
            self.value_log[k] = []
        for k, v in self.actions_log.items():
            self.writer.add_histogram(f"5.Debug/action/{k}", numpy.array(v), counter)
            self.writer.add_scalar(f"5.Debug/action/{k}_mean", numpy.mean(numpy.array(v)), counter)
            debug_log.append(v)
            self.actions_log[k] = []
        for k, v  in debug_params.items():
            self.writer.add_scalar(f"5.Debug/params/{k}", v, counter )
            debug_log.append(v)
        for k, v  in hypermodel_std.items():
            self.writer.add_scalar(f"5.Debug/params/{k}", v, counter )
            debug_log.append(v)
        self.debug_logs.loc[counter] = debug_log
        self.debug_logs.to_csv(self.debug_logs_path, sep="\t", index=False)

    def calculation_std(self, params):
        n = len(params)
        if n == 0:
            return 0
        params = torch.cat(params, dim=1)
        params_mean = torch.mean(params, dim=1, keepdim=True)
        params = params - params_mean
        params_cov = torch.mm(params, params.t()) / (n - 1)
        params_std = torch.sum(torch.diag(params_cov))
        return params_std.item()
