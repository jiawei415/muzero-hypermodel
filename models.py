import torch
import numpy
from functools import reduce
from abc import ABC, abstractmethod


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


def products(inputs):
    return reduce(lambda x, y: x * y, inputs)


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


class MuZeroNetwork:
    def __new__(cls, config):
        return MuZeroFullyConnectedNetwork(config)


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
        self.value_hyper, self.reward_hyper, self.state_hyper = config.hypermodel
        self.init_norm, self.target_norm, self.prior_model = {}, {}, {}
        self.splited_shapes, self.splited_sizes = {}, {}
        self.config = config

        if not self.config.use_representation and self.config.stacked_observations == 0:
            encoding_size = self.config.observation_shape[-1]
        else:
            representation_input_size = products(observation_shape) * (stacked_observations + 1) + products(observation_shape[1:]) * stacked_observations
            self.representation_network = LinearBasedNet([representation_input_size] + fc_representation_layers + [encoding_size])

        state_layers = [encoding_size + self.action_space_size] + fc_dynamics_layers + [encoding_size]
        if self.state_hyper:
            print(f"use dynamics state hypermodel!")
            if config.use_last_layer:
                self.state_basedmodel = LinearBasedNet(state_layers[:-1], prior=config.use_priormodel, output_activation=torch.nn.ELU)
                self.state_hypermodel = LinearHyperNet(config.hyper_inp_dim, state_layers[-2:], config.use_priormodel, prior_std=config.prior_model_std)
            else:
                self.state_basedmodel = None
                self.state_hypermodel = LinearHyperNet(config.hyper_inp_dim, state_layers, config.use_priormodel, prior_std=config.prior_model_std)
        else:
            self.dynamics_encoded_state_network = LinearBasedNet(state_layers)

        reward_layers = [encoding_size] + fc_reward_layers + [self.full_support_size]
        if self.reward_hyper:
            print(f"use dynamics reward hypermodel!")
            if config.use_last_layer:
                self.reward_basedmodel = LinearBasedNet(reward_layers[:-1], prior=config.use_priormodel, output_activation=torch.nn.ELU)
                self.reward_hypermodel = LinearHyperNet(config.hyper_inp_dim, reward_layers[-2:], config.use_priormodel, prior_std=config.prior_model_std)
            else:
                self.reward_basedmodel = None
                self.reward_hypermodel = LinearHyperNet(config.hyper_inp_dim, reward_layers, config.use_priormodel, prior_std=config.prior_model_std)
        else:
            self.dynamics_reward_network = LinearBasedNet(reward_layers)

        value_layers = [encoding_size] + fc_value_layers + [self.full_support_size]
        if self.value_hyper:
            print(f"use prediction value hypermodel!")
            if config.use_last_layer:
                self.value_basedmodel = LinearBasedNet(value_layers[:-1], prior=config.use_priormodel, output_activation=torch.nn.ELU)
                self.value_hypermodel = LinearHyperNet(config.hyper_inp_dim, value_layers[-2:], config.use_priormodel, prior_std=config.prior_model_std)
            else:
                self.value_basedmodel = None
                self.value_hypermodel = LinearHyperNet(config.hyper_inp_dim, value_layers, config.use_priormodel, prior_std=config.prior_model_std)
        else:
            self.prediction_value_network = LinearBasedNet(value_layers)

        self.prediction_policy_network = LinearBasedNet([encoding_size] + fc_policy_layers + [self.action_space_size])

    def representation(self, observation):
        encoded_state, _ = self.representation_network(
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
        policy_logits, _ = self.prediction_policy_network(encoded_state)
        if self.value_hyper:
            hidden_out, prior_hidden_out = self.value_basedmodel(encoded_state) if self.value_basedmodel else (encoded_state, None)
            value, value_params = self.value_hypermodel(noise_z, hidden_out, prior_hidden_out)
        else:
            value, _ = self.prediction_value_network(encoded_state)
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
            hidden_out, prior_hidden_out = self.state_basedmodel(x) if self.state_basedmodel else (x, None)
            next_encoded_state, state_params = self.state_hypermodel(noise_z, hidden_out, prior_hidden_out)
        else:
            next_encoded_state, _ = self.dynamics_encoded_state_network(x)
            state_params = None

        if self.reward_hyper:
            hidden_out, prior_hidden_out = self.reward_basedmodel(next_encoded_state) if self.reward_basedmodel else (next_encoded_state, None)
            reward, reward_params = self.reward_hypermodel(noise_z, hidden_out, prior_hidden_out)
        else:
            reward, _ = self.dynamics_reward_network(next_encoded_state)
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

    def get_hypermodel_std(self,):
        hypermodel_std = dict()
        hypermodel_std["value_params.weight"] = \
            self.calculation_std(self.value_hypermodel.hypermodel.weight) if self.value_hyper else 0
        hypermodel_std["reward_params.weight"] = \
            self.calculation_std(self.reward_hypermodel.hypermodel.weight) if self.reward_hyper else 0
        hypermodel_std["state_params.weight"] = \
            self.calculation_std(self.state_hypermodel.hypermodel.weight) if self.state_hyper else 0
        return hypermodel_std

    def calculation_std(self, params):
        std = torch.sum(torch.diag(torch.mm(params, params.t()))).cpu()
        return std.detach().numpy()

    def debug(self, noise_z):
        value_params = self.value_hypermodel.gen_params(noise_z).t() if self.value_hyper else None
        state_params = self.state_hypermodel.gen_params(noise_z).t() if self.state_hyper else None
        reward_params = self.reward_hypermodel.gen_params(noise_z).t() if self.reward_hyper else None
        return {"value_params": value_params, "reward_params": reward_params, "state_params": state_params,}


class LinearBasedNet(torch.nn.Module):
    def __init__(
        self,
        layers,
        bias=True,
        prior=False,
        output_activation=torch.nn.Identity,
        activation=torch.nn.ELU
    ) -> None:
        super().__init__()
        assert len(layers) > 0

        self.prior = prior
        based_layers, prior_layers = [], []
        for i in range(len(layers) - 1):
            act = activation if i < len(layers) - 2 else output_activation
            based_layers += [torch.nn.Linear(layers[i], layers[i + 1], bias=bias), act()]
            if self.prior:
                prior_layers += [torch.nn.Linear(layers[i], layers[i + 1], bias=bias), act()]
        self.basedmodel =  torch.nn.Sequential(*based_layers)
        if self.prior:
            self.priormodel = torch.nn.Sequential(*prior_layers)
            for param in self.priormodel.parameters():
                param.requires_grad = False

    def forward(self, x):
        out = self.basedmodel(x)
        prior_out = self.priormodel(x) if self.prior else None
        return out, prior_out


class LinearPriorNet(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            prior_mean: float or numpy.ndarray = 0.,
            prior_std: float or numpy.ndarray = 1.,
    ):
        super().__init__()

        self.in_features, self.out_features = input_size, output_size
        # (fan-out, fan-in)
        self.weight = numpy.random.randn(output_size, input_size).astype(numpy.float32)
        self.weight = self.weight / numpy.linalg.norm(self.weight, axis=1, keepdims=True)

        if isinstance(prior_mean, numpy.ndarray):
            self.bias = prior_mean
        else:
            self.bias = numpy.ones(output_size, dtype=numpy.float32) * prior_mean

        if isinstance(prior_std, numpy.ndarray):
            if prior_std.ndim == 1:
                assert len(prior_std) == output_size
                self.prior_std = numpy.diag(prior_std).astype(numpy.float32)
            elif prior_std.ndim == 2:
                assert prior_std.shape == (output_size, output_size)
                self.prior_std = prior_std
            else:
                raise ValueError
        else:
            assert isinstance(prior_std, (float, int, numpy.float32, numpy.int32, numpy.float64, numpy.int64))
            self.prior_std = numpy.eye(output_size, dtype=numpy.float32) * prior_std

        self.weight = torch.nn.Parameter(torch.from_numpy(self.prior_std @ self.weight).float())
        self.bias = torch.nn.Parameter(torch.from_numpy(self.bias).float())

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = torch.nn.functional.linear(x, self.weight.to(x.device), self.bias.to(x.device))
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, not self.bias.sum() == 0
        )


class LinearHyperNet(torch.nn.Module):
    def __init__(
        self,
        noise_dim: int,
        layers: list or tuple,
        prior: bool = True,
        prior_mean: float or numpy.ndarray = 0.,
        prior_std: float or numpy.ndarray = 1.,
    ):
        super().__init__()

        self.prior = prior
        self.splited_shapes, self.splited_sizes = self.gen_shape_size(layers)
        self.in_features = noise_dim
        self.out_features = sum(self.splited_sizes)
        self.hypermodel = torch.nn.Linear(self.in_features, self.out_features)
        if self.prior:
            self.priormodel = LinearPriorNet(self.in_features, self.out_features, prior_mean, prior_std)

    def gen_shape_size(self, layers):
        shapes, sizes = [], []
        for i in range(len(layers)-1):
            shapes.extend([(layers[i], layers[i+1]), (1, layers[i+1])])
            sizes.extend([layers[i] * layers[i+1], layers[i+1]])
        return shapes, sizes

    def split_params(self, params):
        shapes, sizes = self.splited_shapes, self.splited_sizes
        params = params.split(sizes, dim=1)
        params_splited = []
        for param, shape in zip(params, shapes):
            params_splited.append(param.reshape((-1,) + shape))
        return params_splited

    def base_forward(self, x, params):
        splited_params = self.split_params(params)
        x = x.unsqueeze(dim=1)
        for i in range(0, len(splited_params), 2):
            x = torch.bmm(x, splited_params[i]) + splited_params[i+1]
            if i != len(splited_params) - 2:
                x = torch.nn.functional.relu(x)
        return x.squeeze(dim=1)

    def forward(self, z, x, prior_x=None):
        params = self.hypermodel(z)
        out = self.base_forward(x, params)
        if prior_x is not None and self.prior:
            prior_params = self.priormodel(z)
            prior_out = self.base_forward(prior_x, prior_params)
            out += prior_out
        return out, params

    def regularization(self, z, p=2):
        params = self.hypermodel(z)
        reg_loss = torch.norm(params, dim=1, p=p).square()
        return reg_loss.mean()

    def gen_params(self, z):
        return self.hypermodel(z)
