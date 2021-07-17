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

game_name = "mountaincar"
tb_data_path = f"./results/{game_name}/"
labels = {"+hyper": "hypermodel", "+prior": "priormodel", "+normal": "normalization", "+target": "use_loss_noise", "+reg": "reg_loss"}
xs = dict()
y1s = dict()
for root, dirs, files in os.walk(tb_data_path): 
    for name in dirs:
        print(os.path.join(root, name))
    if len(files) != 0:
        label = "muzero"
        config = pd.read_csv(os.path.join(root, files[1]), sep="\t")
        for k, v in labels.items():
            conf = eval(config[config.key == v].value.to_list()[0])
            if isinstance(conf, list) and 1 in conf:
                label += k
                if len(conf) == 3:
                    v, r, s = conf
                    if v == 1: label += '_v'
                    if r == 1: label += '_r'
                    if s == 1: label += '_s'
                elif len(conf) == 2:
                    v, r = conf
                    if v == 1: label += '_v'
                    if r == 1: label += '_r'    
            elif isinstance(conf, bool) and conf:
                label += k

        tb_data = event_accumulator.EventAccumulator(os.path.join(root, files[0])) 
        tb_data.Reload()
        keys = tb_data.scalars.Keys()
        for key in keys:
            if "2.TestPlayer/1.Total_reward" in key:
                y1s[label] = []
                for item in tb_data.scalars.Items(key):
                    y1s[label].append(item.value)
            elif "3.Workers/2.Played_steps" in key:
                xs[label] = []
                for item in tb_data.scalars.Items(key):
                    xs[label].append(item.value)

wanted1 = ['muzero', 'muzero+hyper_r']
wanted2 = ['muzero', 'muzero+hyper_r', 'muzero+hyper_r+prior_r', 'muzero+hyper_r+normal_r', 'muzero+hyper_r+target_r', 'muzero+hyper_r+reg']
wanted3 = ['muzero', 'muzero+hyper_r', 'muzero+hyper_r+prior_r+normal_r', 'muzero+hyper_r+prior_r+target_r', 'muzero+hyper_r+prior_r+reg']
wanted4 = ['muzero', 'muzero+hyper_r', 'muzero+hyper_r+prior_r+normal_r+target_r+reg']

# wanted1 = ['muzero', 'muzero+hyper_v']
# wanted2 = ['muzero', 'muzero+hyper_v', 'muzero+hyper_v+prior_v', 'muzero+hyper_v+normal_v', 'muzero+hyper_v+target_v', 'muzero+hyper_v+reg']
# wanted3 = ['muzero', 'muzero+hyper_v', 'muzero+hyper_v+prior_v+normal_v', 'muzero+hyper_v+prior_v+target_v', 'muzero+hyper_v+prior_v+reg']
# wanted4 = ['muzero', 'muzero+hyper_v', 'muzero+hyper_v+prior_v+normal_v+target_v+reg']

# wanted = wanted2
for i, wanted in enumerate([wanted1, wanted2, wanted3, wanted4]):
    for label in wanted:
        if label not in xs.keys():
            continue
        x = np.array(xs[label])
        y = np.array(y1s[label])
        smoothed_y = smooth(y, 0.6)
        plt.plot(x, smoothed_y, label=label, linewidth=3)
        # plt.fill_between(x, r1, r2, alpha=0.5)

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
        plt.ylim(-550, 0)
        # plt.ylim(-0.2, 1.2)
        plt.xlim(-1, max(x))
        plt.xlabel('sample num', fontsize=20)
        plt.ylabel('total reward', fontsize=20)

    plt.title(f"{game_name}")
    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.3), loc=8, borderaxespad=0, fontsize=16,)
    plt.tight_layout()
    plt.savefig(f"./figures/{game_name}_{i}")
    plt.show()