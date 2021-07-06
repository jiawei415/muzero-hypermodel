import numpy
import torch
import importlib
import pandas as pd
from utils import MCTS, GameHistory

class Debug:
    def __init__(self, model, config, writer):
        self.config = config
        game_module = importlib.import_module("games." + self.config.game_filename)
        Game = game_module.Game
        self.game = Game(self.config.seed)
        self.model = model
        self.writer = writer
        self.noise_dim = int(self.config.hyper_inp_dim)
        self.debug_noise = torch.normal(0, 1, [1, self.noise_dim]) * self.config.normal_noise_std   
        
        keys = []
        for i in self.game.legal_actions():
            keys.extend([f"mcts_action_{i}", f"model_action_{i}"])
        keys.extend(["value_params", "reward_params", "state_params"])
        self.debug_logs_path = self.config.results_path + "/debug_logs.csv"
        self.debug_logs = pd.DataFrame(columns=keys)
        self.debug_logs.to_csv(self.debug_logs_path, sep="\t", index=False)

    def start_debug(self, counter):
        observation = self.game.reset()
        noise_z = numpy.random.normal(0, 1, [1, self.noise_dim]) * self.config.normal_noise_std
        game_history = GameHistory()
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        with torch.no_grad():
            stacked_observations = game_history.get_stacked_observations(
                -1,
                self.config.stacked_observations,
            )
            # Choose the action
            root, mcts_info = MCTS(self.config).run(
                noise_z,
                self.config.num_simulations,
                self.model,
                stacked_observations,
                self.game.legal_actions(),
                self.game.to_play(),
                True,
            )
            debug_observations = (
                torch.tensor(stacked_observations)
                .float()
                .unsqueeze(0)
                .to(next(self.model.parameters()).device)
            )
            _, _, debug_logits, _, _ = self.model.initial_inference(
                    debug_observations, torch.tensor(noise_z, dtype=torch.float)
                )
            debug_policy = torch.softmax(debug_logits, dim=1).squeeze()
            debug_params = self.model.debug(self.debug_noise)
            self.debug_log(root, debug_policy, debug_params, counter)
        self.game.close()

    def debug_log(self, root, debug_policy, debug_params, counter):
        debug_log = []
        for i in self.config.action_space:
            self.writer.add_scalar(f"5.Debug/mcts_action{i}", root.children[i].prior, counter)
            self.writer.add_scalar(f"5.Debug/model_action{i}", debug_policy[i], counter)
            debug_log.extend([root.children[i].prior, debug_policy[i].numpy()])
        for name, param in debug_params.items():
            if param is not None:
                if self.config.save_histogram_log: self.writer.add_histogram(f"5.Debug/{name}", param)
                self.writer.add_scalar(f"5.Debug/{name}", torch.std(param), counter )
                debug_log.append(torch.std(param).detach().numpy())
            else:
                debug_log.append(0)
        self.debug_logs.loc[counter] = debug_log
        self.debug_logs.to_csv(self.debug_logs_path, sep="\t", index=False)
