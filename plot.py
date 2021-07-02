import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def smooth(scalar, weight=0.6):
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def plot_curve(logs, use_training_step=False):
    for key, values in logs.items():
        for i, (value, date) in enumerate(zip(values, dates)):
            smoothed_value = smooth(value)
            if not use_training_step:
                plt.plot(smoothed_value, label=f'{key}_{date}')
                plt.xticks(range(len(smoothed_value)), range(len(smoothed_value)))
            else:
                plt.plot( logs["training_step"][i], smoothed_value, label=f'{key}_{date}')
                # plt.plot(smoothed_value, label=f'{key}_{date}')
                # plt.xticks(range(len(smoothed_value)), logs["training_step"][i])
        plt.legend()
        plt.savefig(f'{key}.jpg')
        plt.show()

game_name = "cartpole"
dates = ["20210629234153", "20210629234232"]
experiment_name = []
training_logs_dict = {}
variance_logs_dict = {}
action_logs_dict = {}
for date in dates:
    logs_path = f"/home/ztjiaweixu/Code/tx-muzero-hypermodel/results/{game_name}_{date}"
    
    training_logs = pd.read_csv(f"{logs_path}/training_logs.csv", sep="\t")
    columns = training_logs.columns.tolist()
    for column in columns:
        if column not in training_logs_dict.keys():
            training_logs_dict[column] = [training_logs[column].values]
        else:
            training_logs_dict[column].append(training_logs[column].values)

    variance_logs = pd.read_csv(f"{logs_path}/variance_logs.csv", sep="\t")
    columns = variance_logs.columns.tolist()
    for column in columns:
        if column not in variance_logs_dict.keys():
            variance_logs_dict[column] = [variance_logs[column].values]
        else:
            variance_logs_dict[column].append(variance_logs[column].values)

    action_logs = pd.read_csv(f"{logs_path}/action_logs.csv", sep="\t")
    columns = action_logs.columns.tolist()
    for column in columns:
        if column not in action_logs_dict.keys():
            action_logs_dict[column] = [action_logs[column].values]
        else:
            action_logs_dict[column].append(action_logs[column].values)

# plot_curve(training_logs_dict)
# plot_curve(variance_logs_dict, True)
# plot_curve(action_logs_dict)

from tensorboard.backend.event_processing import event_accumulator
import os
game_name = "deepsea"
tb_data_path = f"./results/{game_name}/"
sample_num = dict()
total_reward = dict()
epiosed_num = dict()
acrobot_reward_label = {
    "20210628192700": "muzero",
    "20210628184554": "hyper",
    "20210628192655": "hyper+normal",
    "20210628191405": "hyper+prior",
    "20210629201540": "hyper+target",
    "20210628192919": "hyper+normal+prior",
    "20210628192049": "hyper+normal+target",
    "20210628192924": "hyper+normal+prior+target",
    "20210628192209": "hyper+normal+prior+tagret+reg"
}
mountaincar_reward_label = {
    "20210629195741": "muzero",
    "20210628194235": "hyper",
    "20210628193115": "hyper+normal",
    "20210628193044": "hyper+prior",
    "20210628194038": "hyper+target",
    "20210628194520": "hyper+normal+prior",
    "20210628190301": "hyper+normal+target",
    "20210628194422": "hyper+normal+prior+target",
    "20210628194459": "hyper+normal+prior+tagret+reg"
}
deepsea_value_label = {
    "20210629200923": "muzero",
    "20210628201322": "hyper",
    "20210628201248": "hyper+normal",
    "20210628201548": "hyper+prior",
    "20210628200538": "hyper+target",
    "20210628202315": "hyper+normal+prior",
    "20210628202124": "hyper+normal+target",
    "20210628201417": "hyper+normal+prior+target",
    "20210628201845": "hyper+normal+prior+tagret+reg"
}
deepsea_reward_label = {
    "20210629200923": "muzero",
    "20210628201119": "hyper",
    "20210628201222": "hyper+normal",
    "20210628201531": "hyper+prior",
    "20210628200514": "hyper+target",
    "20210628202251": "hyper+normal+prior",
    "20210628202047": "hyper+normal+target",
    "20210628201339": "hyper+normal+prior+target",
    "20210628201810": "hyper+normal+prior+tagret+reg"
}

experiment_date = []
for root, dirs, files in os.walk(tb_data_path): 
    for name in dirs:
        print(os.path.join(root, name))
        experiment_date.append(name.split('_')[-1])
    for name in files:
        print(os.path.join(root, name))
        tb_data = event_accumulator.EventAccumulator(os.path.join(root, name)) 
        tb_data.Reload()
        keys = tb_data.scalars.Keys()
        for key in keys:
            # print(f"key: {key}")
            if "Self_played_steps" in key:
                sample_num[root.split('/')[-1]] = []
                for item in tb_data.scalars.Items(key):
                    # print(f"step: {item.step} \t value: {item.value}")
                    sample_num[root.split('/')[-1]].append(item.value)
            elif "Total_reward/1." in key:
                total_reward[root.split('/')[-1]] = []
                epiosed_num[root.split('/')[-1]] = []
                for item in tb_data.scalars.Items(key):
                    # print(f"step: {item.step} \t value: {item.value}")
                    total_reward[root.split('/')[-1]].append(item.value)
                    epiosed_num[root.split('/')[-1]].append(item.step)

wanted1 = ["muzero", "hyper"]
wanted2 = ["hyper", "hyper+normal", "hyper+prior", "hyper+target"]
wanted3 = ["hyper+normal", "hyper+normal+prior", "hyper+normal+target", "hyper+normal+prior+target",]
wanted4 = ["muzero", "hyper+prior"]
labels = deepsea_reward_label
wanteds = [wanted1, wanted2, wanted3, wanted4]
# wanted = wanted1
for i, wanted in enumerate(wanteds):
    plt.figure(figsize=(16,8))
    for x, y, date in zip(sample_num.values(), total_reward.values(), experiment_date):
        if date not in labels.keys():
            print(f"date: {date}")
            continue
        if labels[date] not in wanted:
            continue
        x = np.array(x)
        y = np.array(y)
        smoothed_y = smooth(y, 0.6)
        plt.plot(x, smoothed_y, label=labels[date], linewidth=3)
        # plt.fill_between(x, r1, r2, alpha=0.5)

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
        # plt.ylim(-550, 0)
        plt.ylim(-0.2, 1.2)
        plt.xlim(-1, max(x))
        plt.xlabel('sample num', fontsize=20)
        plt.ylabel('total reward', fontsize=20)

    plt.title(f"{game_name}_reward")
    plt.grid()
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0, fontsize=16,)
    plt.tight_layout()
    plt.savefig(f"figures/{game_name}_reward{i}")
    plt.show()
