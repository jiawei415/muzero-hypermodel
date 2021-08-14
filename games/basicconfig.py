class BasicConfig():
    def __init__(self):
        # Important Config
        # value reward state
        self.hypermodel = [0, 0, 0]
        self.priormodel = [0, 0, 0]
        self.normalization = [0, 0, 0]
        self.target_noise = [0, 0, 0]

        self.use_priormodel = False
        self.use_normalization = False
        self.use_target_noise = False
        self.use_reg_loss = False
        self.use_value_target_noise = False
        
        # Based Config
        self.reg_loss_coef = 1e-4
        self.prior_model_std = 1
        self.normal_noise_std = 1
        self.target_noise_std = 0.1
        self.hyper_inp_dim = 32
        self.num_simulations = 50
        self.reanalyse_num_simulations = 10
        self.target_update_freq = 100
        self.num_unroll_steps = 10
        self.td_steps = 5
        self.train_frequency = 100
        self.train_proportion = 0.1
        self.start_train = 1
        self.train_mode = 1
        self.total_episode = 100
        self.seed = 0

        ### Game
        self.use_reward_wrapper = False
        self.use_custom_env = True
        self.fix_init_state = False
        self.players = list(range(1))  # List of players. You should only edit the length

        # Fully Connected Network
        self.stacked_observations = 0
        self.use_representation = True
        self.encoding_size = 8
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = [16]  # Define the hidden layers in the value network
        self.fc_policy_layers = [16]  # Define the hidden layers in the policy network
        
        # Evaluate
        self.test_times = 5
        self.debug_times = 10
        self.record_video = False
        self.record_frequency = 10
        self.use_mcts = True

        ### Self-Play
        self.max_moves = 500  # Maximum number of moves if game is not finished before
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        ### Training
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 10000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 128  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 100  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.02  # Initial learning rate
        self.lr_decay_rate = 0.9  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000

        ### Replay Buffer
        self.replay_buffer_size = 500  # Number of self-play games to keep in the replay buffer
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_reanalyse = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.all_reanalyse = False
        self.use_multiprocess = True
        self.num_process = 32

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25

    def train_times(self, played_games):
        if played_games <= 10:
            return 100 # 200 if self.use_reanalyse else 100
        elif played_games <= 20:
            return 200 # 400 if self.use_reanalyse else 200
        elif played_games <= 40:
            return 400 # 800 if self.use_reanalyse else 400
        else:
            return 600 # 1000 if self.use_reanalyse else 600

    def train_per_paly(self, played_steps):
        if self.train_mode == 1:
            train_times = int(played_steps * self.train_proportion)
        elif self.train_mode == 2:
            train_times = int(self.train_frequency * self.train_proportion)
        elif self.train_mode == 3:
            train_times = int(self.train_frequency * self.train_proportion)
            self.train_proportion = min(0.1 + self.train_proportion, 5.0)
        return train_times