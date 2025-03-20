import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

import time, sys

import numpy as np
import random


import seaborn as sns

from src.util.replay_memory import ReplayMemory
from src.learning.knapsack_neural_network import *
from src.problem.knapsack.environment.environment import Environment
from src.problem.knapsack.learning.brain_ppo import BrainPPO
from src.problem.knapsack.environment.knapsack import Knapsack

import torch


VALIDATION_SET_SIZE = 100
RANDOM_TRIAL = 100

class TrainerPPO:

    def __init__(self, args):

        self.args = args

        np.random.seed(args.seed)
        # np.random.seed(np.random.randint(0, 10000))
        self.n_episode = args.n_episode
        self.instance_size = self.args.n_item
        self.action_space_size = 2   # Because we begin at a given city
        self.episode_length = self.instance_size
        self.num_node_feats = 8
        self.capacity_ratio = args.capacity_ratio

        self.validation_set = self.generate_dataset(VALIDATION_SET_SIZE, np.random.randint(10000))
        self.len_validation_set = len(self.validation_set)

        #plt.clf()
        #l1, l2 = zip(*sorted(zip(self.validation_set[0].weight_list, self.validation_set[0].profit_list)))
        #plt.plot(l1, l2, 'b.')
        #plt.xticks([], [])
        #plt.yticks([], [])

        #plt.xlabel("Weights")
        #plt.ylabel("Profits")
        #plt.savefig("corr_weakly.png")


        self.plot_training = args.plot_training
        self.save_dir = args.save_dir

        self.brain = BrainPPO(self.args, self.num_node_feats)

        self.memory = ReplayMemory()

        self.timestep = 0
        self.reward_scaling = 0.0001

    def run_training(self):

        avg_best_random = 0.0
        avg_worst_random = 0.0
        avg_avg_random = 0.0
        avg_optimal = 0.0

        for i in range(self.len_validation_set):
            best, worst, avg = self.evaluate_random(i)
            avg_best_random += best / self.reward_scaling
            avg_worst_random += worst / self.reward_scaling
            avg_avg_random += avg / self.reward_scaling
            #avg_optimal += self.evaluate_exact(i)

        avg_best_random = avg_best_random / self.len_validation_set
        avg_worst_random = avg_worst_random / self.len_validation_set
        avg_avg_random = avg_avg_random / self.len_validation_set
        avg_optimal = avg_optimal / self.len_validation_set

        print('[BEST-RANDOM]', avg_best_random)
        print('[AVG-RANDOM]', avg_avg_random)
        print('[WORST-RANDOM]', avg_worst_random)
        print('[OPT]', avg_optimal)

        sys.stdout.flush()

        start_time = time.time()

        if self.plot_training:
            iter_list = []
            reward_list = []

        print('[INFO]', 'iter', 'time', 'avg_reward_learning', 'loss')

        cur_best_reward = -10000000

        for i in range(self.n_episode):

            loss = self.run_episode()

            if i % 100 == 0 or (i % 10 == 0 and i < 101):

                avg_reward = 0.0

                for j in range(self.len_validation_set):
                    avg_reward += self.evaluate_instance(j) / self.reward_scaling

                avg_reward = avg_reward / self.len_validation_set

                cur_time = round(time.time() - start_time, 2)

                print('[DATA]', i, cur_time, avg_reward, loss)
                sys.stdout.flush()

                if self.plot_training:
                    iter_list.append(i)
                    reward_list.append(avg_reward)
                    plt.clf()
                    plt.plot(iter_list, reward_list, linestyle="-", color='y')

                    lowest_random_list = [avg_best_random for _ in range(len(iter_list))]
                    highest_random_list = [avg_worst_random for _ in range(len(iter_list))]

                    plt.plot(iter_list, [avg_avg_random for _ in range(len(iter_list))], label="AVG-RANDOM", linestyle="-.",
                             color='b')
                    plt.fill_between(iter_list, lowest_random_list, highest_random_list, alpha=0.4)
                    plt.plot(iter_list, reward_list, linestyle="-", label="DQN", color='y')
                    plt.legend(loc=3)
                    out_file = '%s/log_training_curve_reward.png' % self.save_dir
                    plt.savefig(out_file)
                    sns.set_theme()
                    plt.clf()

                fn = "iter_%d_model.pth.tar" % i

                if avg_reward >= cur_best_reward:
                    cur_best_reward = avg_reward
                    self.brain.save(folder=self.args.save_dir, filename=fn) # Record only the improved model
                elif i % 10000 == 0:
                    self.brain.save(folder=self.args.save_dir, filename=fn) # Still record it  eah 10000 it.

    def run_episode(self):

        instance = Knapsack.generate_random_instance(n_item=self.instance_size, lb=1, ub=10000,
                                                     ratio=self.capacity_ratio, cor_type=self.args.correlation)
        env = Environment(instance)

        cur_state = env.get_initial_environment()

        for i in range(self.episode_length):

            self.timestep += 1

            nn_input = env.make_nn_input(cur_state, self.args.mode)
            avail = env.get_valid_actions(cur_state)

            available_tensor = torch.FloatTensor(avail)

            out_action, log_prob_action, _ = self.brain.policy_old.act(nn_input, available_tensor)

            action = out_action.item()
            cur_state, reward = env.get_next_state_with_reward(cur_state, action)

            self.memory.add_sample(nn_input, out_action, log_prob_action, reward, cur_state.is_done(), available_tensor)

            if self.timestep % self.args.update_timestep == 0:
                self.brain.update(self.memory)
                self.memory.clear_memory()
                self.timestep = 0

    def evaluate_instance(self, idx, random_selection=False):

        instance = self.validation_set[idx]
        env = Environment(instance)
        cur_state = env.get_initial_environment()

        total_reward = 0

        for i in range(self.episode_length):

            nn_input = env.make_nn_input(cur_state, self.args.mode)
            avail = env.get_valid_actions(cur_state)

            available_tensor = torch.FloatTensor(avail)

            if random_selection:
                avail_idx = np.argwhere(avail == 1).reshape(-1)
                out_action = random.choice(avail_idx)
            else:
                out_action, _, _ = self.brain.policy_old.act(nn_input, available_tensor)

            action = out_action.item()

            cur_state, reward = env.get_next_state_with_reward(cur_state, action)

            total_reward += reward

        return total_reward

    def generate_dataset(self, size, seed):

        dataset = []
        for i in range(size):
            new_instance = Knapsack.generate_random_instance(n_item=self.instance_size, lb=1, ub=10000,
                                                             ratio=self.capacity_ratio, cor_type=self.args.correlation,
                                                             seed=seed)
            dataset.append(new_instance)
            seed += 1

        return dataset

    def evaluate_random(self, idx):
        best = 0.
        worst = 100000.
        avg = 0.

        for _ in range(RANDOM_TRIAL):
            cost = self.evaluate_instance(idx, random_selection=True)
            best = max(best, cost)
            worst = min(worst, cost)
            avg += cost
        return best, worst, avg / RANDOM_TRIAL