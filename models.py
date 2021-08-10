import torch
import numpy
from abc import ABC, abstractmethod

class MuZeroNetwork:
    def __new__(cls, config):
        return MuZeroFullyConnectedNetwork(config)


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


class AbstractNetwork(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)


class MuZeroFullyConnectedNetwork(AbstractNetwork):
    def __init__(self, config):
        super().__init__()
        observation_shape = config.observation_shape
        stacked_observations = config.stacked_observations
        encoding_size = config.encoding_size
        fc_reward_layers = config.fc_reward_layers
        fc_value_layers = config.fc_value_layers
        fc_policy_layers = config.fc_policy_layers
        fc_representation_layers = config.fc_representation_layers
        fc_dynamics_layers = config.fc_dynamics_layers
        self.action_space_size = len(config.action_space)
        self.full_support_size = 2 * config.support_size + 1
        self.value_prior, self.reward_prior, self.state_prior = config.priormodel
        self.value_hyper, self.reward_hyper, self.state_hyper = config.hypermodel
        self.value_normal, self.reward_normal, self.state_normal = config.normalization
        self.init_norm, self.target_norm = {}, {}
        self.config = config

        if not self.config.use_representation and self.config.stacked_observations == 0:
            encoding_size = self.config.observation_shape[-1]
        else:
            self.representation_network = torch.nn.DataParallel(
                mlp(
                    observation_shape[0]
                    * observation_shape[1]
                    * observation_shape[2]
                    * (stacked_observations + 1)
                    + stacked_observations * observation_shape[1] * observation_shape[2],
                    fc_representation_layers,
                    encoding_size,
                )
            )
        if self.state_hyper:
            print(f"use dynamics state hypermodel!")
            layers = [encoding_size + self.action_space_size] + fc_dynamics_layers + [encoding_size]
            self.state_shapes, self.state_sizes = self.gen_shape_size(layers)
            state_params_inp_dim = config.hyper_inp_dim
            state_params_out_dim = sum(self.state_sizes)
            self.state_params = torch.nn.Linear(state_params_inp_dim, state_params_out_dim)
            if self.state_prior:
                self.state_prior_params = self.gen_prior_params(state_params_inp_dim, state_params_out_dim)
            if self.state_normal:
                self.init_norm['state'], self.target_norm['state'] = [], []
        else:
            self.dynamics_encoded_state_network = torch.nn.DataParallel(
                mlp(
                    encoding_size + self.action_space_size,
                    fc_dynamics_layers,
                    encoding_size,
                )
            )

        if self.reward_hyper:
            print(f"use dynamics reward hypermodel!")
            layers = [encoding_size] + fc_reward_layers + [self.full_support_size]
            self.reward_shapes, self.reward_sizes = self.gen_shape_size(layers)
            reward_params_inp_dim = config.hyper_inp_dim
            reward_params_out_dim = sum(self.reward_sizes)
            self.reward_params = torch.nn.Linear(reward_params_inp_dim, reward_params_out_dim)
            if self.reward_prior:
                self.reward_prior_params = self.gen_prior_params(reward_params_inp_dim, reward_params_out_dim)
            if self.reward_normal:
                self.init_norm['reward'], self.target_norm['reward'] = [], []
        else:
            self.dynamics_reward_network = torch.nn.DataParallel(
                mlp(encoding_size, fc_reward_layers, self.full_support_size)
            )

        self.prediction_policy_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_policy_layers, self.action_space_size)
        )

        if self.value_hyper:
            print(f"use prediction value hypermodel!")
            layers = [encoding_size] + fc_value_layers + [self.full_support_size]
            self.value_shapes, self.value_sizes = self.gen_shape_size(layers)
            value_params_inp_dim = config.hyper_inp_dim
            value_params_out_dim = sum(self.value_sizes)
            self.value_params = torch.nn.Linear(value_params_inp_dim, value_params_out_dim)
            if self.value_prior:
                self.value_prior_params = self.gen_prior_params(value_params_inp_dim, value_params_out_dim)
            if self.value_normal:
                self.init_norm['value'], self.target_norm['value'] = [], []
        else:
            self.prediction_value_network = torch.nn.DataParallel(
                mlp(encoding_size, fc_value_layers, self.full_support_size)
            )

    def gen_shape_size(self, layers):
        shapes = []
        sizes = []
        for i in range(len(layers)-1):
            shapes.extend([(layers[i], layers[i+1]), (1, layers[i+1])])
            sizes.extend([layers[i] * layers[i+1], layers[i+1]])
        return shapes, sizes

    def gen_prior_params(self, inp_dim, out_dim):
        std = self.config.prior_model_std
        normal_deviates = numpy.random.standard_normal((out_dim, inp_dim)) * std
        radius = numpy.linalg.norm(normal_deviates, axis=1, keepdims=True)
        prior_B = normal_deviates / radius
        prior_D = numpy.eye(out_dim)
        prior_params = torch.from_numpy(prior_D.dot(prior_B)).float()
        return torch.nn.parameter.Parameter(prior_params.T, requires_grad=False)
    
    def representation(self, observation):
        encoded_state = self.representation_network(
            observation.view(observation.shape[0], -1)
        )
        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def prediction(self, encoded_state, noise_z):
        policy_logits = self.prediction_policy_network(encoded_state)
        if self.value_hyper:
            value_params = self.value_params(noise_z)
            if self.value_prior:
                value_prior_params = torch.mm(noise_z, self.value_prior_params.to(noise_z.device))
                value_params_ = value_params + value_prior_params
            else:
                value_params_ = value_params
            split_params = self.split_params(value_params_, "value")
            if self.value_normal:
                if len(self.init_norm['value']) == 0:
                    self.gen_norm(split_params, "value")
                split_params = self.get_normal_params(split_params, "value")
            value = self.basemodel_forward(encoded_state, split_params)
        else:
            value = self.prediction_value_network(encoded_state)
            value_params = None
        return policy_logits, value, value_params
 
    def dynamics(self, encoded_state, action, noise_z):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size))
            .to(action.device)
            .float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        if self.state_hyper:
            state_params = self.state_params(noise_z)
            if self.state_prior:
                state_prior_params = torch.mm(noise_z, self.state_prior_params.to(noise_z.device))
                state_params_ = state_params + state_prior_params
            else:
                state_params_ = state_params
            split_params = self.split_params(state_params_, "state")
            if self.state_normal:
                if len(self.init_norm['state']) == 0:
                    self.gen_norm(split_params, "state")
                split_params = self.get_normal_params(split_params, "state")
            next_encoded_state = self.basemodel_forward(x, split_params)
        else:
            next_encoded_state = self.dynamics_encoded_state_network(x)
            state_params = None

        if self.reward_hyper:
            reward_params = self.reward_params(noise_z)
            if self.reward_prior:
                reward_prior_params = torch.mm(noise_z, self.reward_prior_params.to(noise_z.device))
                reward_params_ = reward_params + reward_prior_params
            else:
                reward_params_ = reward_params
            split_params = self.split_params(reward_params_, "reward")
            if self.reward_normal:
                if len(self.init_norm['reward']) == 0:
                    self.gen_norm(split_params, "reward")
                split_params = self.get_normal_params(split_params, "reward")
            reward = self.basemodel_forward(next_encoded_state, split_params)
        else:
            reward = self.dynamics_reward_network(next_encoded_state)
            reward_params = None

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state

        return next_encoded_state_normalized, reward, state_params, reward_params

    def basemodel_forward(self, inputs, params):
        inputs = inputs.unsqueeze(dim=1)
        for i in range(0, len(params), 2):
            inputs = torch.bmm(inputs, params[i]) + params[i+1]
            if i != len(params) - 2:
                inputs = torch.nn.functional.relu(inputs)
        return inputs.squeeze(dim=1)

    def debug(self, noise_z):
        value_params = self.value_params(noise_z).t() if self.value_hyper else None
        state_params = self.state_params(noise_z).t() if self.state_hyper else None
        reward_params = self.reward_params(noise_z).t() if self.reward_hyper else None

        return {"value_params": value_params, "reward_params": reward_params, "state_params": state_params, }

    def initial_inference(self, observation, noise_z):
        if not self.config.use_representation and self.config.stacked_observations == 0:
            encoded_state = observation.reshape(-1, self.config.observation_shape[-1])
        else:
            encoded_state = self.representation(observation)
            
        policy_logits, value, value_params = self.prediction(encoded_state, noise_z)
        # reward equal to 0 for consistency
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )

        return value, reward, policy_logits, encoded_state, value_params

    def recurrent_inference(self, encoded_state, action, noise_z):
        next_encoded_state, reward, state_params, reward_params = self.dynamics(encoded_state, action, noise_z)
        policy_logits, value, value_params = self.prediction(next_encoded_state, noise_z)
        return value, reward, policy_logits, next_encoded_state, value_params, state_params, reward_params

    def split_params(self, params, hyper_type):
        if hyper_type == "reward":
            sizes = self.reward_sizes
            shapes = self.reward_shapes
        elif hyper_type == "state":
            sizes = self.state_sizes
            shapes = self.state_shapes
        elif hyper_type == "value":
            sizes = self.value_sizes
            shapes = self.value_shapes
        params = params.split(sizes, dim=1)
        params_splited = []
        for param, shape in zip(params, shapes):
            params_splited.append(param.reshape((-1,) + shape))
        return params_splited

    def get_normal_params(self, params, normal_type):
        init_norm = self.init_norm[normal_type]
        target_norm = self.target_norm[normal_type]
        gain = 1.
        for i, param in enumerate(params):
            if param.shape[1] == 1:
                continue
            param *= gain * target_norm[i] / init_norm[i]
        return params

    def gen_norm(self, params, norm_type):
        print(f"gen {norm_type} norm!")
        init_norm, target_norm = [], []
        for param in params:
            init_norm.append(torch.norm(param).detach().numpy())
            target_norm.append(torch.norm(torch.nn.init.xavier_normal_(
                torch.empty(size=param.size()))).detach().numpy())
        self.init_norm[norm_type] = init_norm
        self.target_norm[norm_type] = target_norm

    def get_hypermodel(self,):
        hypermodel_std = dict()
        hypermodel_std["value_params.weight"] = \
            self.calculation_std(self.value_params.weight) if self.value_hyper else 0
        hypermodel_std["reward_params.weight"] = \
            self.calculation_std(self.reward_params.weight) if self.reward_hyper else 0
        hypermodel_std["state_params.weight"] = \
            self.calculation_std(self.state_params.weight) if self.state_hyper else 0
        return hypermodel_std

    def calculation_std(self, params):
        std = torch.sum(torch.diag(torch.mm(params, params.t()))).cpu()
        return std.detach().numpy()


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)
