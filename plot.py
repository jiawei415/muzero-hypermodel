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
    return smoothed

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

plot_curve(training_logs_dict)
plot_curve(variance_logs_dict, True)
plot_curve(action_logs_dict)