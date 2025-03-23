import torch
import numpy as np
from types import SimpleNamespace

from utils.read_tsptw import read_rl

import dgl
from rl_agent.hybrid_cp_rl_solver.problem.tsptw.environment.tsptw import TSPTW
from rl_agent.hybrid_cp_rl_solver.architecture.graph_attention_network import GATNetwork
from rl_agent.hybrid_cp_rl_solver.problem.tsptw.learning.actor_critic import ActorCritic

# This code is adapted from hybrid-cp-rl-solver by Cappart et al. [AAAI 2021]
# https://github.com/qcappart/hybrid-cp-rl-solver/blob/master/src/problem/tsptw/solving/solver_binding.py

class TSPTWAgent(object):

    def __init__(self, load_folder, instance_path, n_city, grid_size, 
                 max_tw_gap, max_tw_size, seed, rl_algorithm):
        """
        :param load_folder: folder where the pytorch model (.pth.tar) is saved
        :param instance  : instance file name (.txt)
        
        """

        self.n_city = n_city
        self.grid_size = grid_size
        self.max_tw_gap = max_tw_gap
        self.max_tw_size = max_tw_size
        self.rl_algorithm = rl_algorithm
        self.seed = seed

        self.max_dist = np.sqrt(self.grid_size ** 2 + self.grid_size ** 2)
        self.max_tw_value = (self.n_city - 1) * (self.max_tw_size + self.max_tw_gap)

        self.instance_path =  instance_path
        n_city, travel_time, x_coord, y_coord, time_windows = read_rl(instance_path)
        self.instance = TSPTW(n_city, travel_time, x_coord, y_coord, time_windows)

        self.load_folder = load_folder
        self.model_file, self.latent_dim, self.hidden_layer, self.n_node_feat, self.n_edge_feat = \
            self.find_model()

        self.edge_feat_tensor = self.instance.get_edge_feat_tensor(self.max_dist)

        self.input_graph = self.initialize_graph()

        if self.rl_algorithm == "dqn":
            embedding = [(self.n_node_feat, self.n_edge_feat),
                            (self.latent_dim, self.latent_dim),
                            (self.latent_dim, self.latent_dim),
                            (self.latent_dim, self.latent_dim)]

            self.model = GATNetwork(embedding, self.hidden_layer, self.latent_dim, 1)
            self.model.load_state_dict(torch.load(self.model_file, map_location='cpu'), strict=True)
            self.model.eval()

        elif self.rl_algorithm == "ppo":

            # reproduce the NameSpace of argparse
            args = SimpleNamespace(latent_dim=self.latent_dim, hidden_layer=self.hidden_layer)
            self.actor_critic_network = ActorCritic(args, self.n_node_feat, self.n_edge_feat)

            self.actor_critic_network.load_state_dict(torch.load(self.model_file, map_location='cpu'), strict=True)
            self.model = self.actor_critic_network.action_layer
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
                    elif line[1] == "n_node_feat:":
                        n_node_feat = int(line[2].strip())
                    elif line[1] == "n_edge_feat:":
                        n_edge_feat = int(line[2].strip())

                if '[DATA]' in line:
                    line = line.split(' ')
                    it = int(line[1].strip())
                    reward = float(line[3].strip())
                    if reward > best_reward:
                        best_reward = reward
                        best_it = it

        assert best_it >= 0, "No model found"
        model_str = '%s/iter_%d_model.pth.tar' % (self.load_folder, best_it)
        return model_str, latent_dim, hidden_layer, n_node_feat, n_edge_feat

    def initialize_graph(self):
        """
        Return and initialize a graph corresponding to the first state of the DP model.
        :return: the graph
        """

        g = dgl.from_networkx(self.instance.graph)

        node_feat = [[self.instance.x_coord[i] / self.grid_size,
                      self.instance.y_coord[i] / self.grid_size,
                      self.instance.time_windows[i][0] / self.max_tw_value,
                      self.instance.time_windows[i][1] / self.max_tw_value,
                      0,
                      1,
                      ]
                     for i in range(g.number_of_nodes())]

        node_feat_tensor = torch.FloatTensor(node_feat).reshape(g.number_of_nodes(), self.n_node_feat)

        g.ndata['n_feat'] = node_feat_tensor
        g.edata['e_feat'] = self.edge_feat_tensor
        batched_graph = dgl.batch([g])

        return batched_graph