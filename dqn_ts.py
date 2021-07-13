import os
import gym
import torch
import random
import pprint
import argparse
import datetime
import importlib
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import DQNPolicy
from tianshou.utils import BasicLogger
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer

class Game():
    def __init__(self, game_name):
        game_module = importlib.import_module("games." + game_name)
        self.config = game_module.MuZeroConfig()
        self.env = game_module.Game().env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def seed(self, seed):
        self.env.seed(seed)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if self.config.use_reward_wrapper and not self.config.use_custom_env:
            reward = self.reward_wrapper(reward)
        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        return observation

    def close(self):
        self.env.close()

    def reward_wrapper(self, reward):
        reward_ = 0 if reward==-1 else 1
        return reward_

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='acrobot')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.5)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--step-per-epoch', type=int, default=100000)
    parser.add_argument('--step-per-collect', type=int, default=100)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int,
                        nargs='*', default=[128, 128, 128, 128])
    parser.add_argument('--dueling',
                        action="store_true", default=True)
    parser.add_argument('--dueling-q-hidden-sizes', type=int,
                        nargs='*', default=[128, 128])
    parser.add_argument('--dueling-v-hidden-sizes', type=int,
                        nargs='*', default=[128, 128])
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='results')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--prioritized-replay',
                        action="store_true", default=False)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args

def test_dqn(args):
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = DummyVectorEnv(
        [lambda: Game(args.task) for _ in range(args.training_num)])
    test_envs = DummyVectorEnv(
        [lambda: Game(args.task) for _ in range(args.test_num)])
    args.state_shape = train_envs.observation_space[0].shape or train_envs.observation_space[0].n
    args.action_shape = train_envs.action_space[0].shape or train_envs.action_space[0].n
    # seed
    setup_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    if args.dueling:
        hidden_sizes = args.hidden_sizes[:1]
        Q_param = {"hidden_sizes": args.dueling_q_hidden_sizes}
        V_param = {"hidden_sizes": args.dueling_v_hidden_sizes}
        net = Net(args.state_shape, args.action_shape,
              hidden_sizes=hidden_sizes, device=args.device,
              dueling_param=(Q_param, V_param)).to(args.device)
    else:
        net = Net(args.state_shape, args.action_shape,
                hidden_sizes=args.hidden_sizes, device=args.device
                ).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = DQNPolicy(
        net, optim, args.gamma, args.n_step,
        target_update_freq=args.target_update_freq)
    # buffer
    if args.prioritized_replay:
        buf = PrioritizedVectorReplayBuffer(
            args.buffer_size, buffer_num=len(train_envs),
            alpha=args.alpha, beta=args.beta)
    else:
        buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))
    # collector
    train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    log_path = os.path.join(args.logdir, args.task + "_" + os.path.basename(__file__)[:-3] + "_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    writer = SummaryWriter(log_path)
    logger = BasicLogger(writer)
    summary = str(net).replace("\n", " \n\n")
    writer.add_text("Model summary", summary)
    hp_table = []
    for key, value in args.__dict__.items():
        hp_table.extend([f"| {key} | {value} |"])
    writer.add_text("Hyperparameters", "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def train_fn(epoch, env_step):
        # eps annnealing, just a demo
        if env_step <= 100000:
            policy.set_eps(args.eps_train)
        elif env_step <= 500000:
            eps = args.eps_train - (env_step - 100000) / \
                400000 * (0.5 * args.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.5 * args.eps_train)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.step_per_collect, args.test_num,
        args.batch_size, update_per_step=args.update_per_step, train_fn=train_fn,
        test_fn=test_fn, save_fn=save_fn, logger=logger)
    pprint.pprint(result)
    
    # Let's watch its performance!
    env = Game(args.task)
    policy.eval()
    policy.set_eps(args.eps_test)
    collector = Collector(policy, env)
    result = collector.collect(n_episode=1, render=args.render)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews.mean()}, length: {lens.mean()}")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    test_dqn(get_args())
