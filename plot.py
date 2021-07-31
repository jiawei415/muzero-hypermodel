import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

played_step, training_step = {}, {}
test_reward, value_params_std, reward_params_std, state_params_std = {}, {}, {}, {}
total_loss, value_loss, reward_loss, policy_loss = {}, {}, {}, {}
mcts_action_0, mcts_action_1, mcts_action_2 = {}, {}, {}
mcts_action_mean_0, mcts_action_mean_1, mcts_action_mean_2 = {}, {}, {}
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

game_name = "deepsea"
time_tag = "2021073102"
log_path = f"./results/{game_name}/{time_tag}"
labels = {"+hyper": "hypermodel", "+prior": "priormodel", "+normal": "normalization", "+target": "target_noise", "+reg": "use_reg_loss"}

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
        # tb_data = event_accumulator.EventAccumulator(os.path.join(root, files[2]))
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

def plot_scalar(xs, ys, ylabel, weight):
    for i, wanted in enumerate(wanteds):
        plot_flag = False
        for label in wanted:
            if label not in xs.keys():
                continue
            plot_flag = True
            x = xs[label][-1]
            y = ys[label][-1]
            smoothed_y = smooth(y, weight)
            plt.plot(x, smoothed_y, label=label, linewidth=3)
            plt.xlabel('sample num')
            plt.ylabel(ylabel)

        if plot_flag:
            plt.title(f"{game_name}_seed{seed}")
            plt.grid()
            # plt.legend(bbox_to_anchor=(0.5, -0.5), loc=8, borderaxespad=0, fontsize=16,)
            plt.legend(loc='upper left', handlelength=5, borderpad=1.2, labelspacing=1.2)
            plt.tight_layout()
            # plt.savefig(f"./figures/{game_name}_seed{seed}_{i}")
            plt.show()

def plot_action(xs, action_datas):
    for i, wanted in enumerate(wanteds):
        for label in wanted:
            if label not in action_datas["mcts_action_0"].keys():
                continue
            for action, action_data in action_datas.items():
                if action_data != {}:
                    x = xs[label][-1]
                    y = np.array(action_data[label][2])
                    y_min = np.array(action_data[label][0])
                    y_max = np.array(action_data[label][1])
                    plt.plot(x, y, label=action)
                    plt.fill_between(x, y_min, y_max, alpha=0.9)

            plt.xlabel("smaple num")
            plt.ylabel("action probability")
            plt.title(f"{game_name}_seed{seed}_{label}")
            plt.legend(loc='upper left', handlelength=5, borderpad=1.2, labelspacing=1.2)
            plt.tight_layout()
            # plt.savefig(f"./figures/{game_name}_seed{seed}_{label}")
            plt.show()

def plot_distribution(xs, ys, ylabel, weight=0.6, plot_mean=True):
    for i, wanted in enumerate([wanted1, wanted2]):
        for label in wanted:
            if label not in xs.keys():
                continue
            min_len = min([len(y) for y in ys[label]])
            x = xs[label][0][:min_len]
            y, y_min, y_max = gen_ydata(ys[label], min_len, weight)
            plt.plot(x, y, color='#ff7f0e', label=label)
            if plot_mean:
                plt.fill_between(x, y_min, y_max, color='#ff7f0e', alpha=0.9)
            plt.xlabel('smaple num')
            plt.ylabel(ylabel)
            plt.title(f"{game_name}")
            plt.grid()
            plt.legend(loc='upper left', handlelength=5, borderpad=1.2, labelspacing=1.2)
            plt.tight_layout()
            # plt.savefig(f"./figures/{game_name}_{label}")
            plt.show()

def plot_all(xs, action, scalar):
    weight = 0.8
    for i, label in enumerate(wanted1 + wanted2):
        if label not in xs.keys():
            continue
        fig, axes = plt.subplots(len(scalar) + 1, 1, figsize=(10, 10))
        fig.suptitle(f"{game_name}_{label}")
        min_len = min([len(x) for x in xs[label]])
        x = xs[label][0][:min_len]
        for i, (action_name, action_data) in enumerate(action.items()):
            if action_data != {}:
                y, y_min, y_max = gen_ydata(action_data[label], min_len, weight)
                axes[0].plot(x, y, color=COLORS[i], label=action_name)
                axes[0].fill_between(x, y_min, y_max, color=COLORS[i], alpha=0.9)
                axes[0].legend(loc='upper left', handlelength=5, borderpad=1.2, labelspacing=1.2)
                axes[0].set_ylabel('action probability')
            axes[0].set_xlim([min(x)-10, max(x)+10])
            axes[0].grid()
        for i, (ax, (ylabel, ys)) in enumerate(zip(axes[1:], scalar.items())):
            y, y_min, y_max = gen_ydata(ys[label], min_len, weight)
            start = len(x) - len(y) if "loss" in ylabel else 0
            x_ = x[start:]
            ax.plot(x_, y, color=COLORS[i+2])
            ax.fill_between(x_, y_min, y_max, color=COLORS[i+2], alpha=0.9)
            ax.set_xlim([min(x)-10, max(x)+10])
            ax.set_ylabel(ylabel)
            ax.grid()

        plt.xlabel('smaple num')
        plt.subplots_adjust(wspace=0, hspace=0.2)
        plt.savefig(f"./figures/{time_tag}_{game_name}_{label}")
        plt.show()


# COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
COLORS = ['green', 'red','blue', 'orange', 'darkred', 'darkblue', 'black', 'yellow', 'magenta', 'cyan', 'purple', 'pink',
          'brown', 'orange', 'teal', 'lightblue', 'lime', 'lavender', 'turquoise',
          'green', 'tan', 'salmon', 'gold']

for suffix in ["value", "reward", "state"]:
    wanted1 = ['muzero', f'muzero_{suffix}+hyper']
    wanted2 = [f'muzero_{suffix}+hyper', f'muzero_{suffix}+hyper+prior', f'muzero_{suffix}+hyper+normal', f'muzero_{suffix}+hyper+target', f'muzero_{suffix}+hyper+reg']
    wanted3 = [f'muzero_{suffix}+hyper+prior', f'muzero_{suffix}+hyper+prior+normal', f'muzero_{suffix}+hyper+prior+target', f'muzero_{suffix}+hyper+prior+normal+target']
    wanted4 = [f'muzero_{suffix}+hyper+normal', f'muzero_{suffix}+hyper+prior+normal', f'muzero_{suffix}+hyper+normal+target', f'muzero_{suffix}+hyper+prior+normal+target']
    wanted5 = [f'muzero_{suffix}+hyper+target', f'muzero_{suffix}+hyper+prior+target', f'muzero_{suffix}+hyper+normal+target', f'muzero_{suffix}+hyper+prior+normal+target']
    wanted6 = ['muzero', f'muzero_{suffix}+hyper', f'muzero_{suffix}+hyper+prior+normal+target+reg']
    wanteds = [wanted1, wanted2, wanted3, wanted4, wanted5]

    if "value" in suffix: params_std = value_params_std
    if "reward" in suffix: params_std = reward_params_std
    if "state" in suffix: params_std = state_params_std

    scalars = {
        "total reward": test_reward,
        f"{suffix} variance": params_std,
        "value loss": value_loss,
        "reward loss": reward_loss,
    }
    plot_all(played_step, action_mean_datas, scalars)

    # plot_scalar(played_step, test_reward, 'total reward', 0.9)
    # plot_scalar(played_step, params_std, f"{suffix} variance", 0.6)
    # plot_scalar(played_step, value_loss, "value loss", 0.6)
    # plot_action(played_step, action_datas)

    # plot_distribution(played_step, test_reward, "total reward", 0.9)
    # plot_distribution(played_step, params_std, f"{suffix} variance", 0.6)
    # plot_distribution(played_step, value_loss, "value loss")

    # plot_distribution(played_step, test_reward, "total reward", 0.9, False)
    # plot_distribution(played_step, params_std, f"{suffix} variance", 0.6, False)
    # plot_distribution(played_step, value_loss, "value loss", 0.6, False)
