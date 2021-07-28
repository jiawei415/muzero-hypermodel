import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def smooth(scalar, weight=0.6):
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def plot(x, y, label):
    x = np.array(x)
    y = np.array(y)
    smoothed_y = smooth(y, 0.6)
    plt.plot(x, smoothed_y, label=label, linewidth=3)
    plt.title(f"{game_name}")
    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.3), loc=8, borderaxespad=0, fontsize=16,)
    plt.tight_layout()
    plt.show()

game_name = "deepsea"
action = 2
log_path = f"./results/{game_name}/"
labels = {"+hyper": "hypermodel", "+prior": "priormodel", "+normal": "normalization", "+target": "target_noise", "+reg": "use_reg_loss"}
played_step, training_step = {}, {}
test_reward, value_params_std, reward_params_std, state_params_std = {}, {}, {}, {}
mcts_action_0, mcts_action_1, mcts_action_2 = {}, {}, {}
mcts_action_mean_0, mcts_action_mean_1, mcts_action_mean_2 = {}, {}, {}
player_datas = {
    "played_steps": played_step,
    "training_steps": training_step,
    "test_total_reward": test_reward
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

for root, dirs, files in os.walk(log_path): 
    for name in dirs:
        print(os.path.join(root, name))
    if len(files) != 0:
        files = sorted(files)
        label = "muzero"
        config = pd.read_csv(os.path.join(root, files[0]), sep="\t")
        seed = config[config.key == "seed"].value.to_list()[0]
        # label += f"_{seed}"
        v, r, s = eval(config[config.key == "hypermodel"].value.to_list()[0])
        if v == 1: label += '_value'
        if r == 1: label += '_reward'
        if s == 1: label += '_state'
        for k, v in labels.items():
            conf = eval(config[config.key == v].value.to_list()[0])
            if (isinstance(conf, list) and 1 in conf) or (isinstance(conf, bool) and conf):
                label += k
        debug_logs = pd.read_csv(os.path.join(root, files[1]), sep="\t")
        player_logs = pd.read_csv(os.path.join(root, files[3]), sep="\t")
        trainer_logs = pd.read_csv(os.path.join(root, files[4]), sep="\t")
        for k, v in player_datas.items():
            if label in v.keys():
                v[label].append(player_logs[k].to_numpy())
            else:
                v[label] = [player_logs[k].to_numpy()]
        for k, v in debug_datas.items():
            if label in v.keys():
                v[label].append(debug_logs[k].to_numpy())
            else:
                v[label] = [debug_logs[k].to_numpy()]
            # debug_logs[k].plot(label=label)
            # plt.show()
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
        # for k, v in action_datas.items():
        #     if k in debug_logs.columns:
        #         if label in v.keys():
        #             v[label].append([])
        #         else:
        #             v[label] = [[]]
        #         for prob in debug_logs[k]:
        #             data = np.array(eval(prob))
        #             v[label][-1].append(np.mean(data))
        # tb_data = event_accumulator.EventAccumulator(os.path.join(root, files[0])) 
        # tb_data.Reload()
        # keys = tb_data.scalars.Keys()
        # for key in keys:
        #     if "2.TestPlayer/1.Total_reward" in key:
        #         y1s[label] = []
        #         for item in tb_data.scalars.Items(key):
        #             y1s[label].append(item.value)
        #     elif "3.Workers/2.Played_steps" in key:
        #         xs[label] = []
        #         for item in tb_data.scalars.Items(key):
        #             xs[label].append(item.value)

def plot_scalar(xs, ys, ylabel):
    for i, wanted in enumerate(wanteds):
        plt.figure(figsize=(6, 4))
        for label in wanted:
            if label not in xs.keys():
                continue
            x = xs[label][0]
            y = ys[label][0]
            smoothed_y = smooth(y)
            plt.plot(x, smoothed_y, label=label, linewidth=3)

            plt.xlabel('sample num')
            plt.ylabel(ylabel)

        plt.title(f"{game_name}_{seed}")
        plt.grid()
        # plt.legend(bbox_to_anchor=(0.5, -0.5), loc=8, borderaxespad=0, fontsize=16,)
        plt.legend(loc='upper left', handlelength=5, borderpad=1.2, labelspacing=1.2)
        plt.tight_layout()
        # plt.savefig(f"./figures/{game_name}_{i}")
        plt.show()

def plot_action(action_datas):
    for i, wanted in enumerate(wanteds):
        for label in wanted:
            if label not in action_datas["mcts_action_0"].keys():
                continue
            for action, action_data in action_datas.items():
                if action_data != {}:
                    y = np.array(action_data[label][2])
                    x = range(0, len(y) * 100, 100)
                    y_min = np.array(action_data[label][0])
                    y_max = np.array(action_data[label][1])
                    plt.plot(x, y, label=action)
                    plt.fill_between(x, y_min, y_max, alpha=0.9)
        
            plt.xlabel("played step")
            plt.ylabel("action probability")
            plt.title(f"{game_name}_seed{seed}_{label}")
            plt.legend(loc='upper left', handlelength=5, borderpad=1.2, labelspacing=1.2)
            plt.tight_layout()
            # plt.savefig(f"./figures/{game_name}_seed{seed}_{title}")
            plt.show()

def plot_reward(xs, ys):
    for i, wanted in enumerate([wanted1, wanted2]):
        for label in wanted:
            if label not in xs.keys():
                continue
            plt.figure(figsize=(6, 4))
            min_len = min([len(y) for y in ys[label]])
            y_matrix = np.vstack([y[:min_len] for y in ys[label]])
            y_min = smooth(np.min(y_matrix, axis=0), 0.9)
            y_max= smooth(np.max(y_matrix, axis=0), 0.9)
            y = smooth(np.mean(y_matrix, axis=0), 0.9)
            x = xs[label][0][:min_len]
            plt.plot(x, y, color='#1f77b4', label=label)
            plt.fill_between(x, y_min, y_max, color='#1f77b4', alpha=0.9)

            plt.xlabel('smaple num')
            plt.ylabel('total reward')
            plt.title(f"{game_name}")
            plt.grid()
            plt.legend(loc='upper left', handlelength=5, borderpad=1.2, labelspacing=1.2)
            plt.tight_layout()
            # plt.savefig(f"./figures/1_{game_name}_{label}")
            plt.show()

def plot_params(xs, ys):
    for i, wanted in enumerate([wanted1, wanted2]):
        for label in wanted:
            if label not in xs.keys():
                continue
            plt.figure(figsize=(6, 4))
            min_len = min([len(y) for y in ys[label]])
            y_matrix = np.vstack([y[:min_len] for y in ys[label]])
            y_min = smooth(np.min(y_matrix, axis=0))
            y_max= smooth(np.max(y_matrix, axis=0))
            y = smooth(np.mean(y_matrix, axis=0))
            x = xs[label][0][:min_len]
            plt.plot(x, y, color='#ff7f0e', label=label)
            plt.fill_between(x, y_min, y_max, color='#ff7f0e', alpha=0.9)

            plt.xlabel('smaple num')
            plt.ylabel('param variance')
            plt.title(f"{game_name}")
            plt.grid()
            plt.legend(loc='upper left', handlelength=5, borderpad=1.2, labelspacing=1.2)
            plt.tight_layout()
            # plt.savefig(f"./figures/2_{game_name}_{label}")
            plt.show()

def plot_reward_and_params(xs, ys1, ys2):
    smooth_weight = 0.8
    for i, label in enumerate(wanted1 + wanted2):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        fig.suptitle(f"{game_name}_{label}")
        if label not in xs.keys():
            continue
        min_len = min([len(y) for y in ys1[label]])
        x = xs[label][0][:min_len]
        y1_matrix = np.vstack([y[:min_len] for y in ys1[label]])
        y1_min = smooth(np.min(y1_matrix, axis=0), smooth_weight)
        y1_max= smooth(np.max(y1_matrix, axis=0), smooth_weight)
        y1 = smooth(np.mean(y1_matrix, axis=0), smooth_weight)
        
        lns1 = ax1.plot(x, y1, color='#1f77b4', label='reward')
        ax1.fill_between(x, y1_min, y1_max, color='#1f77b4', alpha=0.9)
        ax1.set_ylabel('total reward')
        ax1.grid()

        y2_matrix = np.vstack([y[:min_len] for y in ys2[label]])
        y2_min = smooth(np.min(y2_matrix, axis=0), smooth_weight)
        y2_max= smooth(np.max(y2_matrix, axis=0), smooth_weight)
        y2 = smooth(np.mean(y2_matrix, axis=0), smooth_weight)

        lns2 = ax2.plot(x, y2, color='#ff7f0e', label='variance')
        ax2.fill_between(x, y2_min, y2_max, color='#ff7f0e', alpha=0.9)
        ax2.set_ylabel('param variance')
        ax2.grid()
        # lns = lns1+lns2
        # labs = [l.get_label() for l in lns]
        # fig.legend(lns, labs, loc='upper left', handlelength=5, borderpad=1.2, labelspacing=1.2)
        plt.xlabel('smaple num')
        # plt.savefig(f"./figures/0_{game_name}_{label}")
        plt.show()

def plot_reward_and_params_and_action(xs, ys1, ys2, ys3):
    smooth_weight = 0.8
    for i, label in enumerate(wanted1 + wanted2):
        if label not in xs.keys():
            continue
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
        fig.suptitle(f"{game_name}_{label}")
        min_len = min([len(x) for x in xs[label]])
        x = xs[label][0][:min_len]

        y1_matrix = np.vstack([y[:min_len] for y in ys1[label]])
        y1_min = smooth(np.min(y1_matrix, axis=0), smooth_weight)
        y1_max= smooth(np.max(y1_matrix, axis=0), smooth_weight)
        y1 = smooth(np.mean(y1_matrix, axis=0), smooth_weight)
        
        ax1.plot(x, y1, color=COLORS[0], label='reward')
        ax1.fill_between(x, y1_min, y1_max, color=COLORS[0], alpha=0.9)
        ax1.set_ylabel('total reward')
        ax1.grid()

        y2_matrix = np.vstack([y[:min_len] for y in ys2[label]])
        y2_min = smooth(np.min(y2_matrix, axis=0), smooth_weight)
        y2_max= smooth(np.max(y2_matrix, axis=0), smooth_weight)
        y2 = smooth(np.mean(y2_matrix, axis=0), smooth_weight)

        ax2.plot(x, y2, color=COLORS[1], label='variance')
        ax2.fill_between(x, y2_min, y2_max, color=COLORS[1], alpha=0.9)
        ax2.set_ylabel('param variance')
        ax2.grid()

        for i, (action, action_data) in enumerate(ys3.items()):
            if action_data != {}:
                y3_matrix = np.vstack([y[:min_len] for y in action_data[label]])
                y3_min = smooth(np.min(y3_matrix, axis=0), smooth_weight)
                y3_max= smooth(np.max(y3_matrix, axis=0), smooth_weight)
                y3 = smooth(np.mean(y3_matrix, axis=0), smooth_weight)

                ax3.plot(x, y3, color=COLORS[i+2], label=action)
                ax3.fill_between(x, y3_min, y3_max, color=COLORS[i+2], alpha=0.9)
                ax3.legend(loc='upper left', handlelength=5, borderpad=1.2, labelspacing=1.2)
                ax3.set_ylabel('action probability')
                ax3.grid()

        plt.xlabel('smaple num')
        plt.savefig(f"./figures/{game_name}_{label}")
        plt.show()

def plot_reward_mean(xs, ys):
    for i, wanted in enumerate([wanted1, wanted2]):
        plt.figure(figsize=(6, 4))
        for j, label in enumerate(wanted):
            if label not in xs.keys():
                continue
            min_len = min([len(x) for x in xs[label]])
            x = xs[label][0][:min_len]
            y_matrix = np.vstack([y[:min_len] for y in ys[label]])
            y = smooth(np.mean(y_matrix, axis=0), 0.9)
            plt.plot(x, y, color=COLORS[j], label=label)
            plt.xlabel('smaple num')
            plt.ylabel('total reward')

        plt.title(f"{game_name}")
        plt.grid()
        plt.legend(loc='upper left', handlelength=5, borderpad=1.2, labelspacing=1.2)
        plt.tight_layout()
        # plt.savefig(f"./figures/1_{game_name}")
        plt.show()

def plot_params_mean(xs, ys):
    for i, wanted in enumerate([wanted1, wanted2]):
        plt.figure(figsize=(6, 4))
        for j, label in enumerate(wanted):
            if label not in xs.keys():
                continue
            min_len = min([len(x) for x in xs[label]])
            x = xs[label][0][:min_len]
            y_matrix = np.vstack([y[:min_len] for y in ys[label]])
            y = smooth(np.mean(y_matrix, axis=0))

            plt.plot(x, y, color=COLORS[j], label=label)
            plt.xlabel('smaple num')
            plt.ylabel('param variance')

        plt.title(f"{game_name}")
        plt.grid()
        plt.legend(loc='upper left', handlelength=5, borderpad=1.2, labelspacing=1.2)
        plt.tight_layout()
        # plt.savefig(f"./figures/2_{game_name}")
        plt.show()


suffix = "state"
wanted1 = ['muzero', f'muzero_{suffix}+hyper']
wanted2 = [f'muzero_{suffix}+hyper', f'muzero_{suffix}+hyper+prior', f'muzero_{suffix}+hyper+normal', f'muzero_{suffix}+hyper+target', f'muzero_{suffix}+hyper+reg']
wanted3 = [f'muzero_{suffix}+hyper+prior', f'muzero_{suffix}+hyper+prior+normal', f'muzero_{suffix}+hyper+prior+target', f'muzero_{suffix}+hyper+prior+normal+target']
wanted4 = [f'muzero_{suffix}+hyper+normal', f'muzero_{suffix}+hyper+prior+normal', f'muzero_{suffix}+hyper+normal+target', f'muzero_{suffix}+hyper+prior+normal+target']
wanted5 = [f'muzero_{suffix}+hyper+target', f'muzero_{suffix}+hyper+prior+target', f'muzero_{suffix}+hyper+normal+target', f'muzero_{suffix}+hyper+prior+normal+target']
wanted6 = ['muzero', f'muzero_{suffix}+hyper', f'muzero_{suffix}+hyper+prior+normal+target+reg']
wanteds = [wanted1, wanted2, wanted3, wanted4, wanted5]

# COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
COLORS = ['blue', 'orange', 'green', 'red', 'black', 'yellow', 'magenta', 'cyan', 'purple', 'pink',
          'brown', 'orange', 'teal', 'lightblue', 'lime', 'lavender', 'turquoise',
          'green', 'tan', 'salmon', 'gold', 'darkred', 'darkblue']

if "value" in suffix: params_std = value_params_std
if "reward" in suffix: params_std = reward_params_std
if "state" in suffix: params_std = state_params_std

# plot_scalar(played_step, test_reward, 'total reward')
# plot_reward_mean(played_step, test_reward)
# plot_reward(played_step, test_reward)

# plot_scalar(played_step, params_std, 'params_variance')
# plot_params_mean(played_step, params_std)
# plot_params(played_step, params_std)

# plot_action(action_datas)

# plot_reward_and_params(played_step, test_reward, params_std)
plot_reward_and_params_and_action(played_step, test_reward, params_std, action_mean_datas)