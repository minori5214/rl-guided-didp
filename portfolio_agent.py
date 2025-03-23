import torch
import numpy as np
from types import SimpleNamespace

from utils.read_portfolio import read

from rl_agent.hybrid_cp_rl_solver.problem.portfolio.environment.portfolio import Portfolio
from rl_agent.hybrid_cp_rl_solver.problem.portfolio.learning.actor_critic import ActorCritic
from rl_agent.hybrid_cp_rl_solver.architecture.set_transformer import SetTransformer

class PortfolioAgent(object):

    def __init__(self, load_folder, instance_path, n_item, capacity_ratio,
                 lambda_1, lambda_2, lambda_3, lambda_4,
                 discrete_coeff, seed, rl_algorithm):
        """
        Initialization of the binding
        :param load_folder: folder where the pytorch model (.pth.tar) is saved
        :param n_item: number of item in the portfolio
        :param capacity_ratio: capacity_ratio: capacity of the instance is capacity_ratio * (sum of all the item weights)
        :param lambda_1: coefficient factor for the first moment (mean)
        :param lambda_2: coefficient factor for the second moment (deviation)
        :param lambda_3: coefficient factor for the third moment (skewness)
        :param lambda_4: coefficient factor for the forth moment (kurtosis)
        :param lambda_4: coefficient factor for the forth moment (kurtosis)
        :param lambda_4: coefficient factor for the forth moment (kurtosis)
        :param discrete_coeff: 1 if we take the discretized version of the problem
        :param rl_algorithm: 'ppo' or 'dqn'
        """

        self.n_item = n_item
        self.capacity_ratio = capacity_ratio
        self.moment_factors = [lambda_1, lambda_2, lambda_3, lambda_4]
        self.discrete_coeff = discrete_coeff
        self.rl_algorithm = rl_algorithm
        self.seed = seed

        self.instance_path =  instance_path
        n_item, capacity, weights, means, deviations, skewnesses, kurtosis, moment_factors = \
            read(instance_path)
        self.instance = Portfolio(n_item, capacity, weights, means, deviations, skewnesses, kurtosis, moment_factors)

        self.load_folder = load_folder

        self.model_file, self.latent_dim, self.hidden_layer, self.n_feat = self.find_model()

        self.weigth_norm = np.linalg.norm(self.instance.weights)
        self.mean_norm = np.linalg.norm(self.instance.means)
        self.deviation_norm = np.linalg.norm(self.instance.deviations)
        self.skewnesses_norm = np.linalg.norm(self.instance.skewnesses)
        self.kurtosis_norm = np.linalg.norm(self.instance.kurtosis)

        if self.rl_algorithm == "dqn":

            self.model = SetTransformer(dim_hidden=self.latent_dim, dim_input=self.n_feat, dim_output=2)
            self.model.load_state_dict(torch.load(self.model_file, map_location='cpu'), strict=True)
            self.model.eval()

        elif self.rl_algorithm == "ppo":

            # reproduce the NameSpace of argparse
            args = SimpleNamespace(latent_dim=self.latent_dim, hidden_layer=self.hidden_layer)
            self.actor_critic_network = ActorCritic(args, self.n_feat)
            self.actor_critic_network.load_state_dict(torch.load(self.model_file, map_location='cpu'), strict=True)
            self.model = self.actor_critic_network.action_layer
            self.model.eval()
        elif self.rl_algorithm == "ppoval":

            # reproduce the NameSpace of argparse
            args = SimpleNamespace(latent_dim=self.latent_dim, hidden_layer=self.hidden_layer)
            self.actor_critic_network = ActorCritic(args, self.n_feat)
            self.actor_critic_network.load_state_dict(torch.load(self.model_file, map_location='cpu'), strict=True)
            self.model = self.actor_critic_network.value_layer
            self.model.eval()
        else:
            raise Exception("RL algorithm not implemented")

    def find_model(self):
        """
        Find and return the .pth.tar model for the corresponding instance type.
        :return: the location of the model and some hyperparameters used
        """

        log_file_path = self.load_folder + "/log-training.txt"
        best_reward = 0

        best_it = -1

        with open(log_file_path, 'r') as f:
            for line in f:

                if '[INFO]' in line:
                    line = line.split(' ')
                    if line[1] == "latent_dim:":
                        latent_dim = int(line[2].strip())
                    elif line[1] == "hidden_layer:":
                        hidden_layer = int(line[2].strip())
                    elif line[1] == "n_feat:":
                        n_feat = int(line[2].strip())

                if '[DATA]' in line:
                    line = line.split(' ')
                    it = int(line[1].strip())
                    reward = float(line[3].strip())
                    if reward > best_reward:
                        best_reward = reward
                        best_it = it

        assert best_it >= 0, "No model found"
        model_str = '%s/iter_%d_model.pth.tar' % (self.load_folder, best_it)
        return model_str, latent_dim, hidden_layer, n_feat


    def build_state_feats(self, last_item, cur_weight):
        """
        Build and return the tensor features corresponding to the current state. Must be consistent with the RL environment.
        Especially, mind the normalization constants.
        :param last_item: last item that has been considered
        :param cur_weight: the accumulated weight in the portfolio
        :return: the tensor features
        """

        state_feat = [[self.instance.weights[i] / self.weigth_norm,
                      self.instance.means[i] / self.mean_norm,
                      self.instance.deviations[i] / self.deviation_norm,
                      self.instance.skewnesses[i] / self.skewnesses_norm,
                      self.instance.kurtosis[i] / self.kurtosis_norm,
                      (self.instance.capacity - cur_weight - self.instance.weights[i]) / self.weigth_norm,
                      0 if i < last_item else 1, # Remaining
                      1 if i == last_item else 0,
                      0 if (self.instance.capacity - cur_weight -
                            self.instance.weights[i]) >= 0 else 1] # will exceed
                      for i in range(self.instance.n_item)]

        state_feat_tensor = torch.FloatTensor(state_feat).reshape(self.instance.n_item, self.n_feat)

        return state_feat_tensor