from cp_rl_solver.problem.knapsack.environment.state import State
#from src.problems.knapsack import *

import torch
import numpy as np
import networkx as nx


class Environment:

    def __init__(self, instance, reward_scaling=0.0001):
        """
        Initialize the DP/RL environment
        :param instance: a Knapsack instance
        :param reward_scaling: value for scaling the reward
        """

        self.instance = instance
        self.reward_scaling = reward_scaling

    def get_initial_environment(self):

        weight = 0
        stage = 0

        if self.instance.weight_list[stage] > self.instance.capacity:
            available_action = set([0])
        else:
            available_action = set([0, 1])

        return State(self.instance, weight, stage, available_action)

    def make_nn_input(self, cur_state, mode):

        node_feat = [[self.instance.weight_list[i] / 10000,
                      self.instance.profit_list[i] / 10000,
                      np.log10(self.instance.profit_list[i] / self.instance.weight_list[i]), # Ratio
                      np.log10(self.instance.weight_list[i] / self.instance.profit_list[i]),  # inverse Ratio
                      (self.instance.capacity - cur_state.weight - self.instance.weight_list[i]) / 10000,
                      0 if i < cur_state.stage else 1, # Remaining
                      1 if i == cur_state.stage else 0,
                      0 if (self.instance.capacity - cur_state.weight -
                            self.instance.weight_list[i]) >= 0 else 1] # will exceed
                     for i in range(self.instance.n_item)]
        # Is in missing
        # Remaining capacity

        node_feat_tensor = torch.FloatTensor(node_feat).reshape(self.instance.n_item, 8)
        if mode == 'gpu':
            node_feat_tensor = node_feat_tensor.cuda()

        return node_feat_tensor


    def get_action_size(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return 2


    def get_next_state_with_reward(self, cur_state, action):

        new_state, reward = cur_state.step(action)

        reward = reward * self.reward_scaling

        return new_state, reward

    def get_valid_actions(self, cur_state):
        """
        Input:
            board: current state

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """

        available = np.zeros(self.get_action_size(), dtype=np.int64)
        available_idx = np.array([x for x in cur_state.available_action], dtype=np.int64)
        available[available_idx] = 1
        return available

    def to_string(self, cur_state):
        """
        Input:
            board: current board

        Returns:
             a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return "Stage " + str(cur_state.stage) + ": " + str(cur_state.weight)
