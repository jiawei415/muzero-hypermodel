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
    # plt.fill_between(x, r1, r2, alpha=0.5)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # plt.ylim(-550, 0)
    # plt.xlim(-1, max(x))
    # plt.xlabel('sample num', fontsize=20)
    # plt.ylabel('total reward', fontsize=20)

    plt.title(f"{game_name}")
    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.3), loc=8, borderaxespad=0, fontsize=16,)
    plt.tight_layout()
    plt.show()

game_name = "deepsea"
action = 2
tb_data_path = f"./results/{game_name}/"
labels = {"+hyper": "hypermodel", "+prior": "priormodel", "+normal": "normalization", "+target": "target_noise", "+reg": "use_reg_loss"}
played_step, training_step = {}, {}
test_reward, value_params_std, reward_params_std, state_params_std = {}, {}, {}, {}
mcts_action_0, mcts_action_1, mcts_action_2 = {}, {}, {}
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

for root, dirs, files in os.walk(tb_data_path): 
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
            # v[label] = debug_logs[k].to_numpy()
            # debug_logs[k].plot(label=label)
            # plt.show()
        for k, v in action_datas.items():
            if k in debug_logs.columns:
                v[label] = [[], [], []]
                for prob in debug_logs[k]:
                    data = np.array(eval(prob))
                    v[label][0].append(np.min(data))
                    v[label][1].append(np.max(data))
                    v[label][2].append(np.mean(data))
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

suffix = "value"
wanted1 = ['muzero', f'muzero_{suffix}+hyper']
wanted2 = [f'muzero_{suffix}+hyper', f'muzero_{suffix}+hyper+prior', f'muzero_{suffix}+hyper+normal', f'muzero_{suffix}+hyper+target', f'muzero_{suffix}+hyper+reg']
wanted3 = [f'muzero_{suffix}+hyper+prior', f'muzero_{suffix}+hyper+prior+normal', f'muzero_{suffix}+hyper+prior+target', f'muzero_{suffix}+hyper+prior+normal+target']
wanted4 = [f'muzero_{suffix}+hyper+normal', f'muzero_{suffix}+hyper+prior+normal', f'muzero_{suffix}+hyper+normal+target', f'muzero_{suffix}+hyper+prior+normal+target']
wanted5 = [f'muzero_{suffix}+hyper+target', f'muzero_{suffix}+hyper+prior+target', f'muzero_{suffix}+hyper+normal+target', f'muzero_{suffix}+hyper+prior+normal+target']
wanted6 = ['muzero', f'muzero_{suffix}+hyper', f'muzero_{suffix}+hyper+prior+normal+target+reg']
wanted7 = ['muzero','muzero_value+hyper+normal', 'muzero_reward+hyper']
wanteds = [wanted1, wanted2, wanted3, wanted4, wanted5]
# wanteds = [wanted7]

def plot_scalar(xs, ys, xlabel, ylabel):
    for i, wanted in enumerate(wanteds):
        plt.figure(figsize=(6, 4))
        for label in wanted:
            if label not in xs.keys():
                continue
            x = xs[label][0]
            y = ys[label][0]
            smoothed_y = smooth(y)
            if len(x) != len(y):
                x = range(0, len(y) * 100, 100)
            plt.plot(x, smoothed_y, label=label, linewidth=3)
            # plt.fill_between(x, r1, r2, alpha=0.5)

            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            
            # plt.ylim(-550, 0)
            # # plt.ylim(-0.2, 1.2)
            # plt.xlim(-1, max(x))
            plt.xlabel(xlabel, fontsize=20)
            plt.ylabel(ylabel, fontsize=20)

        plt.title(f"{game_name}_{seed}")
        plt.grid()
        # plt.legend(bbox_to_anchor=(0.5, -0.5), loc=8, borderaxespad=0, fontsize=16,)
        plt.legend(loc='upper left', handlelength=5, borderpad=1.2, labelspacing=1.2)
        plt.tight_layout()
        # plt.savefig(f"./figures/{game_name}_{i}")
        plt.show()

def plot_action(action_datas):
    for i, wanted in enumerate(wanteds):
        for title in wanted:
            if title not in action_datas["mcts_action_0"].keys():
                continue
            # plt.figure(figsize=(6, 4))
            for label, action_data in action_datas.items():
                if action_data != {}:
                    y = np.array(action_data[title][2])
                    x = range(0, len(y) * 100, 100)
                    y_min = np.array(action_data[title][0])
                    y_max = np.array(action_data[title][1])
                    plt.plot(x, y, label=label)
                    plt.fill_between(x, y_min, y_max, alpha=0.9)
        
            plt.xlabel("training step", fontsize=10)
            plt.ylabel("action probability", fontsize=10)
            plt.title(f"{game_name}_{seed}_{title}")
            # plt.legend(bbox_to_anchor=(0.5, -0.5), loc=8, borderaxespad=0, fontsize=16,)
            plt.legend(loc='upper left', handlelength=5, borderpad=1.2, labelspacing=1.2)
            plt.tight_layout()
            plt.savefig(f"./figures/{game_name}_{seed}_{title}")
            plt.show()

def plot_reward(xs, ys):
    for i, wanted in enumerate([wanted1, wanted2]):
        plt.figure(figsize=(6, 4))
        for label in wanted:
            if label not in xs.keys():
                continue
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
            plt.savefig(f"./figures/1_{game_name}_{label}")
            plt.show()

def plot_params(xs, ys):
    for i, wanted in enumerate([wanted1, wanted2]):
        plt.figure(figsize=(6, 4))
        for label in wanted:
            if label not in xs.keys():
                continue
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
            plt.savefig(f"./figures/2_{game_name}_{label}")
            plt.show()

def plot_reward_and_params(xs, ys1, ys2):
    smooth_weight = 0.8
    for i, label in enumerate(wanted1 + wanted2):
        fig = plt.figure(figsize=(6, 4))
        ax1 = fig.add_subplot()
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
        ax1.set_xlabel('smaple num')
        ax1.set_ylabel('total reward')
        
        ax2 = ax1.twinx()
        y2_matrix = np.vstack([y[:min_len] for y in ys2[label]])
        y2_min = smooth(np.min(y2_matrix, axis=0), smooth_weight)
        y2_max= smooth(np.max(y2_matrix, axis=0), smooth_weight)
        y2 = smooth(np.mean(y2_matrix, axis=0), smooth_weight)
        lns2 = ax2.plot(x, y2, color='#ff7f0e', label='variance')
        ax2.fill_between(x, y2_min, y2_max, color='#ff7f0e', alpha=0.9)
        ax2.set_ylabel('param variance')
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper left', handlelength=5, borderpad=1.2, labelspacing=1.2)
        plt.title(f"{game_name}_{label}")
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"./figures/0_{game_name}_{label}")
        plt.show()


# plot_scalar(played_step, test_reward, 'played step', 'total reward')
plot_reward(played_step, test_reward)
if "value" in suffix:
    # plot_scalar(played_step, value_params_std, 'training step', 'value_params_std')
    plot_params(played_step, value_params_std)
    plot_reward_and_params(played_step, test_reward, value_params_std)
if "reward" in suffix:
    # plot_scalar(played_step, reward_params_std, 'training step', 'reward_params_std')
    plot_params(played_step, reward_params_std)
    plot_reward_and_params(played_step, test_reward, reward_params_std)
if "state" in suffix:
    # plot_scalar(played_step, state_params_std, 'training step', 'state_params_std')
    plot_params(played_step, state_params_std)
    plot_reward_and_params(played_step, test_reward, state_params_std)

plot_action(action_datas)