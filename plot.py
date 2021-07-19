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

game_name = "mountaincar"
action = 3
tb_data_path = f"./results/{game_name}/"
labels = {"+hyper": "hypermodel", "+prior": "priormodel", "+normal": "normalization", "+target": "target_noise", "+reg": "use_reg_loss"}
played_step, training_step = {}, {}
test_reward, value_params_std, reward_params_std, state_params_std = {}, {}, {}, {}
player_datas = {
    "played_steps": played_step,
    "training_steps": training_step,
    "test_total_reward": test_reward
}
debug_datas = {
    "value_params.weight": value_params_std,
    "reward_params.weight": reward_params_std,
    "state_params.weight": state_params_std
}

keys = ["test_total_reward", "played_steps", "training_steps", \
        "total_loss", "value_loss", "reward_loss", "policy_loss", \
        "value_params.weight", "reward_params.weight", "state_params.weight"]
for i in range(action):
    keys.extend([f"mcts_action_{i}", f"model_action_{i}"])

for root, dirs, files in os.walk(tb_data_path): 
    for name in dirs:
        print(os.path.join(root, name))
    if len(files) != 0:
        files = sorted(files)
        label = "muzero"
        config = pd.read_csv(os.path.join(root, files[0]), sep="\t")
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
            v[label] = player_logs[k].to_numpy()
        for k, v in debug_datas.items():
            v[label] = debug_logs[k].to_numpy()
            # debug_logs[k].plot(label=label)
            # plt.show()
    
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

suffix = "reward"
wanted1 = ['muzero', f'muzero_{suffix}+hyper']
wanted2 = [f'muzero_{suffix}+hyper', f'muzero_{suffix}+hyper+prior', f'muzero_{suffix}+hyper+normal', f'muzero_{suffix}+hyper+target', f'muzero_{suffix}+hyper+reg']
wanted3 = [f'muzero_{suffix}+hyper+prior', f'muzero_{suffix}+hyper+prior+normal', f'muzero_{suffix}+hyper+prior+target', f'muzero_{suffix}+hyper+prior+normal+target']
wanted4 = [f'muzero_{suffix}+hyper+normal', f'muzero_{suffix}+hyper+prior+normal', f'muzero_{suffix}+hyper+normal+target', f'muzero_{suffix}+hyper+prior+normal+target']
wanted5 = [f'muzero_{suffix}+hyper+target', f'muzero_{suffix}+hyper+prior+target', f'muzero_{suffix}+hyper+normal+target', f'muzero_{suffix}+hyper+prior+normal+target']
wanted6 = ['muzero', f'muzero_{suffix}+hyper', f'muzero_{suffix}+hyper+prior+normal+target+reg']

def plots(xs, ys, xlabel, ylabel):
    for i, wanted in enumerate([wanted1, wanted2, wanted3, wanted4, wanted5, wanted6]):
        for label in wanted:
            if label not in xs.keys():
                continue
            x = xs[label]
            y = ys[label]
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

        plt.title(f"{game_name}")
        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.3), loc=8, borderaxespad=0, fontsize=16,)
        plt.tight_layout()
        # plt.savefig(f"./figures/{game_name}_{i}")
        plt.show()

plots(played_step, test_reward, 'sample num', 'total reward')
plots(training_step, value_params_std, 'training num', 'value_params_std')
plots(training_step, reward_params_std, 'training num', 'reward_params_std')
plots(training_step, state_params_std, 'training num', 'state_params_std')