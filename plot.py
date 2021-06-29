import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

game_name = "cartpole"
dates = ["20210629203630", "20210629213335"]
experiment_name = []
logs_dict = {}
for date in dates:
    logs_path = f"/home/ztjiaweixu/Code/tx-muzero-hypermodel/results/{game_name}_{date}"
    
    training_logs = pd.read_csv(f"{logs_path}/training_logs.csv", sep="\t")
    columns = training_logs.columns.tolist()
    for column in columns:
        if column not in logs_dict.keys():
            logs_dict[column] = [training_logs[column].values]
        else:
            logs_dict[column].append(training_logs[column].values)

    variance_logs = pd.read_csv(f"{logs_path}/variance_logs.csv", sep="\t")
    columns = variance_logs.columns.tolist()
    for column in columns:
        if column not in logs_dict.keys():
            logs_dict[column] = [variance_logs[column].values]
        else:
            logs_dict[column].append(variance_logs[column].values)

for key, values in logs_dict.items():
    for value, date in zip(values, dates):
        plt.plot(value, label=f'{key}_{date}')
    plt.legend()
    plt.savefig(f'{key}.jpg')
    plt.show()