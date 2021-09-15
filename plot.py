import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

played_step, training_step, configs = {}, {}, {}
test_reward, value_params_std, reward_params_std, state_params_std = {}, {}, {}, {}
total_loss, value_loss, reward_loss, policy_loss = {}, {}, {}, {}
mcts_action_0, mcts_action_1, mcts_action_2 = {}, {}, {}
mcts_action_mean_0, mcts_action_mean_1, mcts_action_mean_2 = {}, {}, {}
mcts_value, target_model_value, model_value = {}, {}, {}
mcts_value_mean, target_model_value_mean, model_value_mean = {}, {}, {}
player_datas = {
    "played_steps": played_step,
    "training_steps": training_step,
    "test_total_reward": test_reward
}
loss_datas = {
    "total_loss": total_loss,
    "value_loss": value_loss,
    "reward_loss": reward_loss,
    "policy_loss": policy_loss,
}
debug_datas = {
    "value_params.weight": value_params_std,
    "reward_params.weight": reward_params_std,
    "state_params.weight": state_params_std,
}
action_datas = {
    "mcts_action_0": mcts_action_0,
    "mcts_action_1": mcts_action_1,
    "mcts_action_2": mcts_action_2,
}
action_mean_datas = {
    "mcts_action_mean_0": mcts_action_mean_0,
    "mcts_action_mean_1": mcts_action_mean_1,
    "mcts_action_mean_2": mcts_action_mean_2,
}
value_datas = {
    # "mcts_value": mcts_value,
    "target_model_value": target_model_value,
    "model_value": model_value,
}
value_mean_datas = {
    # "mcts_value": mcts_value_mean,
    "target_model_value": target_model_value_mean,
    "model_value": model_value_mean,
}

def smooth(scalar, weight=0.6):
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def gen_ydata(ys, min_len, weight):
    ymin_len = min([len(y) for y in ys])
    min_len = min(min_len, ymin_len)
    y_matrix = np.vstack([y[:min_len] for y in ys])
    y_min = smooth(np.min(y_matrix, axis=0), weight)
    y_max= smooth(np.max(y_matrix, axis=0), weight)
    y = smooth(np.mean(y_matrix, axis=0), weight)
    return y, y_min, y_max

try:
    time_tag = f"2021{sys.argv[1]}"
except:
    time_tag = "2021091401"
game_name = "deepsea"
action_num = 3
debug_action_history = False
log_path = f"./results/{game_name}/{time_tag}"
titles = {"+hyper": "hypermodel", "+prior": "priormodel", "+normal": "normalization", "+target": "target_noise", "+reg": "use_reg_loss"}

for root, dirs, files in os.walk(log_path):
    for name in dirs:
       print(os.path.join(root, name))
    if len(files) != 0:
        if 'config_logs.csv' not in files:
            continue
        title = "muzero"
        config = pd.read_csv(os.path.join(root, 'config_logs.csv'), sep="\t")
        title += "_p" if eval(config[config.key == "PER"].value.to_list()[0]) else "_np"
        v, r, s = eval(config[config.key == "hypermodel"].value.to_list()[0])
        if v == 1: title += '_value'
        if r == 1: title += '_reward'
        if s == 1: title += '_state'
        for k, v in titles.items():
            conf = eval(config[config.key == v].value.to_list()[0])
            if (isinstance(conf, list) and 1 in conf) or (isinstance(conf, bool) and conf):
                title += k
        # seed = config[config.key == "seed"].value.to_list()[0]
        # td_steps = config[config.key == "td_steps"].value.to_list()[0]
        # value_loss_weight = config[config.key == "value_loss_weight"].value.to_list()[0]
        # num_unroll_steps = config[config.key == "num_unroll_steps"].value.to_list()[0]
        # support_size = config[config.key == "support_size"].value.to_list()[0]
        # prior_model_std = config[config.key == "prior_model_std"].value.to_list()[0]
        use_last_layer = config[config.key == "use_last_layer"].value.to_list()[0] if "use_last_layer" in config['key'].values else False
        base_weight_decay = config[config.key == "base_weight_decay"].value.to_list()[0]
        # label = title + f"\t td_steps: {td_steps} value_loss_weight: {value_loss_weight} num_unroll_steps: {num_unroll_steps} support_size: {support_size} use_last_layer: {use_last_layer} prior_model_std: {prior_model_std} base_weight_decay: {base_weight_decay}"
        play_with_improve = config[config.key == "play_with_improve"].value.to_list()[0] if "play_with_improve" in config['key'].values else False
        learn_with_improve = config[config.key == "learn_with_improve"].value.to_list()[0] if "learn_with_improve" in config['key'].values else False
        search_with_improve = config[config.key == "search_with_improve"].value.to_list()[0] if "search_with_improve" in config['key'].values else False
        label = title + f"\t use_last_layer: {use_last_layer} base_weight_decay: {base_weight_decay} play: {play_with_improve} learn: {learn_with_improve} search: {search_with_improve}"
        if game_name == "deepsea":
            size = config[config.key == "size"].value.to_list()[0]
            deterministic = config[config.key == "deterministic"].value.to_list()[0] if "deterministic" in config['key'].values else True
            randomize_actions = config[config.key == "randomize_actions"].value.to_list()[0] if "randomize_actions" in config['key'].values else False
            label += f" size: {size} deterministic: {deterministic} randomize_actions: {randomize_actions}"
        debug_logs = pd.read_csv(os.path.join(root, 'debug_logs.csv'), sep="\t")
        player_logs = pd.read_csv(os.path.join(root, 'palyer_logs.csv'), sep="\t")
        for k, v in player_datas.items():
            if label in v.keys():
                v[label].append(player_logs[k].to_numpy())
            else:
                v[label] = [player_logs[k].to_numpy()]
        for k, v in loss_datas.items():
            if k in player_logs.columns:
                data = player_logs[k].to_numpy()
                start = np.where(data == 0)[0][-1]
                if label in v.keys():
                    v[label].append(data[start+1:])
                else:
                    v[label] = [data[start+1:]]
        for k, v in debug_datas.items():
            if label in v.keys():
                v[label].append(debug_logs[k].to_numpy())
            else:
                v[label] = [debug_logs[k].to_numpy()]
        for (k, v), (k_mean, v_mean) in zip(action_datas.items(), action_mean_datas.items()):
            if k in debug_logs.columns:
                v[label] = [[], [], []]
                for prob in debug_logs[k]:
                    data = np.array(eval(prob))
                    v[label][0].append(np.min(data))
                    v[label][1].append(np.max(data))
                    v[label][2].append(np.mean(data))
                if label in v_mean.keys():
                    v_mean[label].append(v[label][2])
                else:
                    v_mean[label] = [v[label][2]]
        for (k, v), (k_mean, v_mean) in zip(value_datas.items(), value_mean_datas.items()):
            if k in debug_logs.columns:
                v[label] = [[], [], []]
                for prob in debug_logs[k]:
                    data = np.array(eval(prob))
                    v[label][0].append(np.min(data))
                    v[label][1].append(np.max(data))
                    v[label][2].append(np.mean(data))
                if label in v_mean.keys():
                    v_mean[label].append(v[label][2])
                else:
                    v_mean[label] = [v[label][2]]

def plot_all(xs, value, action, scalar):
    weight = 0.8
    for label in xs.keys():
        fig, axes = plt.subplots(len(scalar) + 2, 1, figsize=(20, 30))
        fig.suptitle(game_name + " " + label.replace("\t", "\n"))
        min_len = min([len(x) for x in xs[label]])
        x = xs[label][0][:min_len]
        for i, (value_name, value_data) in enumerate(value.items()):
            if value_data != {}:
                y, y_min, y_max = gen_ydata(value_data[label], min_len, weight)
                axes[0].plot(x, y, color=COLORS[i], label=value_name)
                axes[0].fill_between(x, y_min, y_max, color=COLORS[i], alpha=0.2)
            axes[0].legend(loc='upper right', handlelength=5, borderpad=1.2, labelspacing=1.2, fontsize=8)
            axes[0].set_ylabel('inital state value')
            axes[0].set_xlim([min(x)-10, max(x)+10])
            axes[0].grid()
        for i, (action_name, action_data) in enumerate(action.items()):
            if action_data != {}:
                y, y_min, y_max = gen_ydata(action_data[label], min_len, weight)
                axes[1].plot(x, y, color=COLORS[i+3], label=action_name)
                axes[1].fill_between(x, y_min, y_max, color=COLORS[i+3], alpha=0.2)
            axes[1].legend(loc='upper left', handlelength=5, borderpad=1.2, labelspacing=1.2)
            axes[1].set_ylabel('action probability')
            axes[1].set_xlim([min(x)-10, max(x)+10])
            axes[1].grid()
        for i, (ax, (ylabel, ys)) in enumerate(zip(axes[2:], scalar.items())):
            y, y_min, y_max = gen_ydata(ys[label], min_len, weight)
            start = len(x) - len(y) if "loss" in ylabel else 0
            x_ = x[start:]
            ax.plot(x_, y, color=COLORS[i+3+action_num])
            ax.fill_between(x_, y_min, y_max, color=COLORS[i+3+action_num], alpha=0.2)
            ax.set_xlim([min(x)-10, max(x)+10])
            ax.set_ylabel(ylabel)
            ax.grid()

        plt.xlabel('smaple num')
        plt.subplots_adjust(wspace=0, hspace=0.2)
        plt.savefig(f"./figures/{time_tag}_{game_name}_{label}.png".replace(": ", "_"))
        plt.show()

# COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
COLORS = ['darkgreen', 'darkred', 'lightblue', 'green', 'red','blue', 'orange', 'darkred', 'darkblue', 'black', 'magenta', 'yellow', 'magenta', 'cyan', 'purple', 'pink',
          'brown', 'orange', 'teal', 'lightblue', 'lime', 'lavender', 'tan']

scalars = {"total reward": test_reward}
if reward_params_std != {}:
    scalars.update({"reward variance": reward_params_std})
if state_params_std != {}:
    scalars.update({"state variance": state_params_std})
if value_loss != {}:
    scalars.update({"value loss": value_loss})
if reward_loss != {}:
    scalars.update({"reward loss": reward_loss})
plot_all(played_step, value_mean_datas, action_mean_datas, scalars)

if debug_action_history:
    for root, dirs, files in os.walk(log_path):
        if len(files) != 0:
            title = "muzero"
            config = pd.read_csv(os.path.join(root, 'config_logs.csv'), sep="\t")
            title += "_p" if eval(config[config.key == "PER"].value.to_list()[0]) else "_np"
            v, r, s = eval(config[config.key == "hypermodel"].value.to_list()[0])
            if v == 1: title += '_value'
            if r == 1: title += '_reward'
            if s == 1: title += '_state'
            for k, v in titles.items():
                conf = eval(config[config.key == v].value.to_list()[0])
                if (isinstance(conf, list) and 1 in conf) or (isinstance(conf, bool) and conf):
                    title += k
            seed = config[config.key == "seed"].value.to_list()[0]
            size = config[config.key == "size"].value.to_list()[0]
            deterministic = config[config.key == "deterministic"].value.to_list()[0] if "deterministic" in config['key'].values else True
            randomize_actions = config[config.key == "randomize_actions"].value.to_list()[0] if "randomize_actions" in config['key'].values else False
            use_last_layer = config[config.key == "use_last_layer"].value.to_list()[0] if "use_last_layer" in config['key'].values else False
            base_weight_decay = config[config.key == "base_weight_decay"].value.to_list()[0] if "base_weight_decay" in config['key'].values else 0.0001
            play_with_improve = config[config.key == "play_with_improve"].value.to_list()[0] if "play_with_improve" in config['key'].values else False
            learn_with_improve = config[config.key == "learn_with_improve"].value.to_list()[0] if "learn_with_improve" in config['key'].values else False
            search_with_improve = config[config.key == "search_with_improve"].value.to_list()[0] if "search_with_improve" in config['key'].values else False
            label = title + f"\t seed: {seed} size: {size} deterministic: {deterministic} randomize_actions: {randomize_actions}"
            label += f" use_last_layer: {use_last_layer} base_weight_decay: {base_weight_decay} play: {play_with_improve} learn: {learn_with_improve} search: {search_with_improve}"
            if not eval(randomize_actions):
                continue
            checkpoint = torch.load(os.path.join(root, 'model_best.checkpoint'))
            action_right = np.diagonal(checkpoint['action_mapping']).astype(np.int32)
            print(f"label: {label}")
            debug_logs = pd.read_csv(os.path.join(root, 'debug_logs.csv'), sep="\t")
            best_score = float('-inf')
            best_actions = np.ones(int(size))
            for item in debug_logs['action_history']:
                now_actions = np.array(eval(item)[0])
                now_score = np.sum(now_actions == action_right)
                if now_score > best_score:
                    best_score = now_score
                    best_actions = now_actions
                    print(f"action_right: {action_right} \nbest_actions: {best_actions} score: {best_score}")
