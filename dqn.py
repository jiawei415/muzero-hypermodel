import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import gym
import random
import datetime
import argparse
import importlib
import collections
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN agent')
    # Common arguments
    parser.add_argument('--game-name', type=str, default="acrobot",
                        help='the id of the gym environment')
    # Algorithm specific arguments
    parser.add_argument('--buffer-size', type=int, default=10000,
                         help='the replay memory buffer size')
    parser.add_argument('--start-e', type=float, default=1,
                        help="the starting epsilon for exploration")
    parser.add_argument('--end-e', type=float, default=0.05,
                        help="the ending epsilon for exploration")
    parser.add_argument('--exploration-fraction', type=float, default=0.8,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    args = parser.parse_args()

game_name = args.game_name
game_module = importlib.import_module("games." + game_name)
config = game_module.MuZeroConfig()

results_path = os.path.join("./results", game_name  + "_dqn_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
writer = SummaryWriter(results_path)
hp_table = [f"| {key} | {value} |" for key, value in config.__dict__.items()]
writer.add_text("Hyperparameters", "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True

# modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py#
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        indexs = np.random.choice(len(self.buffer), n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for index in indexs:
            s, a, r, s_prime, done_mask = self.buffer[index]
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst)

    def __len__(self):
        return len(self.buffer)

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, game):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(np.array(game.env.observation_space.shape).prod(), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, game.env.action_space.n)

    def forward(self, x, device):
        x = torch.Tensor(x).to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope =  (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

game = game_module.Game(config.seed)
game.env.action_space.seed(config.seed)
game.env.observation_space.seed(config.seed)
rb = ReplayBuffer(args.buffer_size)
q_network = QNetwork(game).to(device)
target_network = QNetwork(game).to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=config.lr_init)
loss_fn = nn.MSELoss()

num_played_steps = 0
num_played_games = 0
training_step = 0
total_loss = 0

for counter in range(config.episode):
    obs = game.reset().squeeze()
    total_reward = 0
    done = False
    episode_length = 0
    while not done and episode_length < config.max_moves:
        epsilon = 0.1 # linear_schedule(args.start_e, args.end_e, args.exploration_fraction*args.total_timesteps, global_step)
        if random.random() < epsilon:
            action = game.env.action_space.sample()
        else:
            logits = q_network(obs.reshape((1,)+obs.shape), device)
            action = torch.argmax(logits, dim=1).tolist()[0]  
        next_obs, reward, done = game.step(action)
        next_obs = next_obs.squeeze()   
        rb.put((obs, action, reward, next_obs, done))
        obs = next_obs
        total_reward += reward
        episode_length += 1
        num_played_steps += 1
    num_played_games += 1

    print(f'Counter: {counter}/{config.episode}. Last play reward: {total_reward:.2f}. Training step: {training_step}. Played step: {num_played_steps}. Played games: {num_played_games}',)
    writer.add_scalar("1.Total_reward/1.Total_reward", total_reward, counter,)
    writer.add_scalar("1.Total_reward/3.Episode_length", episode_length, counter,)
    writer.add_scalar("2.Workers/1.Self_played_games", num_played_games, counter,)
    writer.add_scalar("2.Workers/2.Training_steps", training_step, counter)
    writer.add_scalar("2.Workers/3.Self_played_steps", num_played_steps, counter)
    writer.add_scalar(
        "2.Workers/5.Training_steps_per_self_played_step_ratio",
        training_step / max(1, num_played_steps),
        counter,
    )
    writer.add_scalar("2.Workers/6.Learning_rate", optimizer.param_groups[0]["lr"], counter)
    writer.add_scalar("3.Loss/1.Total_weighted_loss", total_loss, counter)

    if num_played_games % 2 == 0:
        train_times = config.train_times(num_played_games)
        # for _ in tqdm(range(train_times)):
        for _ in range(train_times):
            if training_step % config.target_update_freq == 0:
                target_network.load_state_dict(q_network.state_dict())
            s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(config.batch_size)
            with torch.no_grad():
                target_max = torch.max(target_network(s_next_obses, device), dim=1)[0]
                td_target = torch.Tensor(s_rewards).to(device) + config.discount * target_max * (1 - torch.Tensor(s_dones).to(device))
            old_val = q_network(s_obs, device).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()
            total_loss = loss_fn(td_target, old_val)

            lr = config.lr_init * config.lr_decay_rate ** (training_step / config.lr_decay_steps)
            optimizer.param_groups[0]["lr"] = lr
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            training_step += 1

game.close()
writer.close()
