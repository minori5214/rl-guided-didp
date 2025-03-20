import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

import random
import time
import sys
import os

import numpy as np

import torch
import torch.nn as nn

from src.util.prioritized_replay_memory import PrioritizedReplayMemory
from src.problem.knapsack.environment.environment import Environment
from src.problem.knapsack.learning.brain_dqn import BrainDQN
from src.problem.knapsack.environment.knapsack import Knapsack

from torch.distributions import Categorical

MEMORY_CAPACITY = 50000
GAMMA = 1
MAX_EPSILON = 1
MIN_EPSILON = 0.01
STEP_EPSILON = 5000.0
LAMBDA = 0.01  # speed of decay
UPDATE_TARGET_FREQUENCY = 500
VALIDATION_SET_SIZE = 100
RANDOM_TRIAL = 100
lReLu = nn.LeakyReLU(0.2)
MIN_LR = 1e-7
INITIAL_LR = 1e-5
MAX_LR = 1e-3
LR_RAMPUP = 200

MIN_VAL = -10000000


class TrainerDQN:
    """
    Definition of the Trainer DQN for the Knapsack
    """

    def __init__(self, args):
        """
        Initialization of the trainer
        :param args:  argparse object taking hyperparameters and instance  configuration
        """

        self.args = args
        np.random.seed(args.seed)

        # np.random.seed(np.random.randint(0, 10000))
        self.instance_size = self.args.n_item
        self.n_action = 2  # Select the item or not

        self.num_node_feats = 8
        self.num_edge_feats = 1

        self.reward_scaling = 0.0001

        self.validation_set = Knapsack.generate_dataset(size=VALIDATION_SET_SIZE, n_item=self.args.n_item, 
                                                        lb=1, ub=10000, 
                                                        capacity_ratio=self.args.capacity_ratio, 
                                                        cor_type=self.args.correlation, seed=np.random.randint(10000),
                                                        is_integer_instance=False)

        self.len_validation_set = len(self.validation_set)

        self.brain = BrainDQN(self.args, self.num_node_feats, self.num_edge_feats)
        self.memory = PrioritizedReplayMemory(MEMORY_CAPACITY)

        self.steps_done = 0  # used for eps-greedy
        self.init_memory_counter = 0

        if args.n_step == -1:
            self.n_step = self.instance_size
        else:
            self.n_step = args.n_step

        print("***********************************************************")
        print("[INFO] NUMBER OF FEATURES")
        print("[INFO] n_node_feat: %d" % self.num_node_feats)
        print("[INFO] n_edge_feat: %d" % self.num_edge_feats)
        print("***********************************************************")

    def run_training(self):
        """
        Run the main loop for training the model
        """

        avg_best_random = 0.0
        avg_worst_random = 0.0
        avg_avg_random = 0.0
        avg_sum_profit = 0.0
        avg_best_ratio = 0.0

        for i in range(self.len_validation_set):
            best, worst, avg = self.evaluate_random(i)
            avg_best_random += best / self.reward_scaling
            avg_worst_random += worst / self.reward_scaling
            avg_avg_random += avg / self.reward_scaling
            avg_sum_profit += sum(self.validation_set[i].profit_list)
            avg_best_ratio += self.evaluate_best_ratio(i)

        avg_best_random = avg_best_random / self.len_validation_set
        avg_worst_random = avg_worst_random / self.len_validation_set
        avg_avg_random = avg_avg_random / self.len_validation_set
        avg_sum_profit = avg_sum_profit / self.len_validation_set
        avg_best_ratio = avg_best_ratio / self.len_validation_set


        print('[BEST-RANDOM]', avg_best_random)
        print('[AVG-RANDOM]', avg_avg_random)
        print('[WORST-RANDOM]', avg_worst_random)
        print('[BEST-RATIO-INSERTION]', avg_best_ratio)

        sys.stdout.flush()

        start_time = time.time()

        if self.args.plot_training:
            iter_list = []
            reward_list = []
            loss_list = []

        self.initialize_memory()

        print('[INFO]', 'iter', 'time', 'avg_reward_learning', 'loss', "avg_max_probability", "beta")

        cur_best_reward = MIN_VAL

        for i in range(self.args.n_episode):

            loss, (avg_max_probability, beta) = self.run_episode(i)

            # If the time exceeded the time limit, we stop the training
            if time.time() - start_time > self.args.time_limit:
                break

            if (i % 10 == 0 and i < 101) or i % 100 == 0:

                avg_reward = 0.0
                for j in range(self.len_validation_set):
                    avg_reward += self.evaluate_instance(j) / self.reward_scaling

                avg_reward = avg_reward / self.len_validation_set

                cur_time = round(time.time() - start_time, 2)

                print('[DATA]', i, cur_time, avg_reward, loss, avg_max_probability, beta)
                sys.stdout.flush()

                if self.args.plot_training:
                    iter_list.append(i)
                    reward_list.append(avg_reward)
                    plt.clf()

                    plt.plot(iter_list, reward_list, linestyle="-", label="DQN", color='y')

                    lowest_random_list = [avg_best_random for _ in range(len(iter_list))]
                    highest_random_list = [avg_worst_random for _ in range(len(iter_list))]

                    plt.plot(iter_list, [avg_avg_random for _ in range(len(iter_list))], label="AVG-RANDOM",
                                linestyle="-.", color='b')
                    plt.fill_between(iter_list, lowest_random_list, avg_avg_random, alpha=0.4)

                    plt.plot(iter_list, [avg_best_ratio for _ in range(len(iter_list))], label="BEST-RATIO", linestyle="-.",
                                color='g')

                    #plt.plot(iter_list, [avg_opt for _ in range(len(iter_list))], label="OPT", linestyle="-.",
                    #         color='r')

                    plt.legend(loc=3)
                    out_file = '%s/log_training_curve_reward.png' % self.args.save_dir
                    plt.savefig(out_file)


                    loss_list.append(loss)
                    plt.clf()

                    plt.plot(iter_list, loss_list, linestyle="-", label="DQN", color='y')

                    plt.legend(loc=3)
                    out_file = '%s/training_curve_loss.png' % self.args.save_dir
                    plt.savefig(out_file)

                    # Add the new loss and avg reward to the csv file
                    with open('%s/training_curve.csv' % self.args.save_dir, 'a') as f:
                        # If the file is empty, we add the header
                        if os.stat('%s/training_curve.csv' % self.args.save_dir).st_size == 0:
                            f.write("iter,avg_reward,loss\n")
                        f.write("%d,%f,%f\n" % (i, avg_reward, loss))

                fn = "iter_%d_model.pth.tar" % i
                #self.brain.save(folder=self.args.save_dir, filename=fn)

                if avg_reward >= cur_best_reward:
                    cur_best_reward = avg_reward
                    self.brain.save(folder=self.args.save_dir, filename=fn)  # Record only the improved model
                elif i % 1000 == 0:
                    self.brain.save(folder=self.args.save_dir, filename=fn)  # Still record it  each 1000 it.

    def initialize_memory(self):

        while self.init_memory_counter < MEMORY_CAPACITY:
            instance = Knapsack.generate_random_instance(n_item=self.instance_size, lb=1, ub=10000,
                                                         ratio=self.args.capacity_ratio, 
                                                         cor_type=self.args.correlation)
            for _ in range(10):
                self.init_memory_counter += 1
                self.run_random_episode_from_instance(instance)

        print("[INFO] Memory Initialized")

    def run_random_episode_from_instance(self, instance):

        env = Environment(instance)
        cur_state = env.get_initial_environment()

        graph_list = []
        rewards_vector = np.zeros(self.instance_size)
        actions_vector = np.zeros(self.instance_size, dtype=np.int16)
        available_vector = np.zeros((self.instance_size, self.n_action))

        total_reward = 0

        i = 0

        while True:

            graph = env.make_nn_input(cur_state, self.args.mode)

            avail = env.get_valid_actions(cur_state)
            avail_idx = np.argwhere(avail == 1).reshape(-1)

            action = random.choice(avail_idx)

            cur_state, reward = env.get_next_state_with_reward(cur_state, action)

            total_reward += reward

            graph_list.append(graph)
            rewards_vector[i] = reward
            actions_vector[i] = action
            available_vector[i] = avail

            if cur_state.is_done():
                break

            i += 1


        episode_last_idx = i

        for i in range(self.instance_size):

            if i <= episode_last_idx:
                cur_graph = graph_list[i]
                cur_available = available_vector[i]

            else:
                cur_graph = graph_list[episode_last_idx]
                cur_available = available_vector[episode_last_idx]

            if i + self.n_step < self.instance_size:

                next_graph = graph_list[i + self.n_step]
                next_available = available_vector[i + self.n_step]
            else:

                next_graph = torch.FloatTensor(np.zeros((self.instance_size, self.num_node_feats)))  # env.make_nn_input_dgl(cur_state)
                next_available = env.get_valid_actions(cur_state)

                if self.args.mode == 'gpu':
                    next_graph = next_graph.cuda()

            state_features = (cur_graph, cur_available)
            next_state_features = (next_graph, next_available)

            reward = sum(rewards_vector[i:i + self.n_step])
            action = actions_vector[i]

            sample = (state_features, action, reward, next_state_features)

            error = abs(sample[2])

            self.memory.add(error, sample)



        return total_reward

    def run_episode(self, episode_idx):
        """
        Run a single episode for training the model (following DQN algorithm)
        :param episode_idx: the index of the current episode done (without considering the memory initialization)
        :return: the loss and the current beta of the softmax selection
        """

        #  Generate a random instance
        instance = Knapsack.generate_random_instance(n_item=self.args.n_item, 
                                                     lb=1, ub=10000,
                                                     ratio=self.args.capacity_ratio, 
                                                     cor_type=self.args.correlation,
                                                     is_integer_instance=False,
                                                     seed=-1)
        #print("Instance")
        #print(instance.weight_list)
        #print(instance.profit_list)
        #print(instance.capacity)
        #print("")

        env = Environment(instance)
        total_loss = 0

        cur_state = env.get_initial_environment()

        set_list = []
        rewards_vector = np.zeros(self.args.n_item)
        actions_vector = np.zeros(self.args.n_item, dtype=np.int16)
        available_vector = np.zeros((self.args.n_item, self.n_action))

        idx = 0

        #  the current temperature for the softmax selection: increase from 0 to MAX_BETA
        temperature = max(0., min(self.args.max_softmax_beta,
                                  (episode_idx - 1) / STEP_EPSILON * self.args.max_softmax_beta))

        #  execute the episode
        avg_max_prob = []
        while True:

            nn_input = env.make_nn_input(cur_state, self.args.mode)
            avail = env.get_valid_actions(cur_state)

            action, max_prob = self.soft_select_action(nn_input, avail, temperature)
            avg_max_prob.append(max_prob)

            #  each time we do a step, we increase the counter, and we periodically synchronize the target network
            self.steps_done += 1
            if self.steps_done % UPDATE_TARGET_FREQUENCY == 0:
                self.brain.update_target_model()

            cur_state, reward = env.get_next_state_with_reward(cur_state, action)

            set_list.append(nn_input)
            rewards_vector[idx] = reward
            actions_vector[idx] = action
            available_vector[idx] = avail

            if cur_state.is_done():
                break

            idx += 1

        episode_last_idx = idx

        for i in range(self.args.n_item):

            if i <= episode_last_idx:
                cur_set = set_list[i]
                cur_available = available_vector[i]
            else:
                cur_set = set_list[episode_last_idx]
                cur_available = available_vector[episode_last_idx]

            if i + self.n_step < self.args.n_item:
                next_set = set_list[i + self.n_step]

                next_available = available_vector[i + self.n_step]
            else:
                next_set = torch.FloatTensor(np.zeros((self.args.n_item, self.num_node_feats)))
                next_available = env.get_valid_actions(cur_state)

                if self.args.mode == 'gpu':
                    next_set = next_set.cuda()

            #  a state correspond to the graph, with the nodes that we can still visit
            state_features = (cur_set, cur_available)
            next_state_features = (next_set, next_available)

            #  the n-step reward
            reward = sum(rewards_vector[i:i + self.n_step])
            action = actions_vector[i]

            sample = (state_features, action, reward, next_state_features)

            x, y, errors, _ = self.get_targets([(0, sample, 0)])  # feed the memory with the new samples
            error = errors[0]
            step_loss = self.learning()  # learning procedure

            self.memory.add(errors, sample)

            total_loss += step_loss

        return total_loss, (np.mean(avg_max_prob), temperature)

    def evaluate_instance(self, idx):
        """
        Evaluate an instance with the current model
        :param idx: the index of the instance in the validation set
        :return: the reward collected for this instance
        """

        instance = self.validation_set[idx]
        env = Environment(instance, self.reward_scaling)
        cur_state = env.get_initial_environment()

        total_reward = 0
        total_profit = 0
        solution = []

        while True:
            #if idx == 0: print("cur state", cur_state.weight, cur_state.stage, cur_state.available_action)
            graph = env.make_nn_input(cur_state, self.args.mode)

            avail = env.get_valid_actions(cur_state)
            #if idx == 0: print("avail", avail)

            action = self.select_action(graph, avail, idx=idx)
            solution.append(action)
            #if idx == 0: print("action", action)
            total_profit += instance.profit_list[cur_state.stage] * action

            cur_state, reward = env.get_next_state_with_reward(cur_state, action)

            total_reward += reward
            #if idx == 0: print("reward", reward)

            if cur_state.is_done():
                break
        #if idx == 0: print("total_profit", total_profit)
        #if idx == 0: print("")
        if idx==0: print("idx={}, solution={}".format(idx, solution))
        return total_reward

    def evaluate_random(self, idx):
        instance = self.validation_set[idx]
        best = 0.
        worst = 100000.
        avg = 0.

        for _ in range(RANDOM_TRIAL):
            reward = self.run_random_episode_from_instance(instance)
            best = max(best, reward)
            worst = min(worst, reward)
            avg += reward
        return best, worst, avg / RANDOM_TRIAL

    def evaluate_best_ratio(self, idx):
        instance = self.validation_set[idx]
        if idx == 0: print(instance.profit_list)
        if idx == 0: print(instance.weight_list)
        if idx == 0: print(instance.capacity)
        if idx == 0: print([p/w for w, p in zip(instance.weight_list, instance.profit_list)])

        acc_profit = 0
        acc_weight = 0

        object_list_sorted = sorted(instance.object_list, key=lambda tup: tup[2]/tup[1], reverse=True)
        if idx == 0: print(object_list_sorted)

        for item in object_list_sorted:

            if acc_weight + item[1] <= instance.capacity:
                acc_weight += item[1]
                acc_profit += item[2]
                if idx == 0: print("item {} is taken".format(item[0]))

        if idx == 0: print("profit", acc_profit)
        return acc_profit

    """
    def evaluate_exact(self, idx):

        instance = self.validation_set[idx]

        solver = pywrapknapsack_solver.KnapsackSolver(
            pywrapknapsack_solver.KnapsackSolver.
                KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')

        solver.Init(instance.profit_list, [instance.weight_list], [instance.capacity])

        opt_value = solver.Solve()


        return opt_value
    """

    def select_action(self, set_input, available, idx=100):
        """
        Select an action according the to the current model
        :param set_input: the featurized set of item (first part of the state)
        :param available: the vector of available (second part of the state)
        :return: the action, following the greedy policy with the model prediction
        """

        batched_set = set_input.unsqueeze(0)
        available = available.astype(bool)
        out = self.brain.predict(batched_set, target=False).squeeze(0)
        #if idx == 0: print("out", out)

        action_idx = np.argmax(out[available])
        #if idx == 0: print("action_idx", action_idx)

        action = np.arange(len(out))[available][action_idx]
        #if idx == 0: print("action", action)

        return action

    def masked_softmax(self, vector: torch.Tensor, mask: torch.Tensor, dim: int = -1,
                        memory_efficient: bool = False, mask_fill_value: float = -1e32, temperature = 1) -> torch.Tensor:

        if mask is None:
            result = torch.nn.functional.softmax(vector, dim=dim)
        else:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            if not memory_efficient:
                # To limit numerical errors from large vector elements outside the mask, we zero these out.
                result = torch.nn.functional.softmax((vector/temperature) * mask, dim=dim)
                result = result * mask
                result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            else:
                masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
                result = torch.nn.functional.softmax(masked_vector/temperature, dim=dim)
        return result

    def soft_select_action(self, set_input, available, beta):
        """
        Select an action according the to the current model with a softmax selection of temperature beta
        :param set_input: the featurized set of item (first part of the state)
        :param available: the vector of available (second part of the state)
        :param beta: the current temperature
        :return: the action, following the softmax selection with the model prediction
        """

        # WARNING: select_action assumes that graph is NOT a batched_graph!!!
        batched_set = set_input.unsqueeze(0)
        available = available.astype(bool)
        out = self.brain.predict(batched_set, target=False)[0].reshape(-1)
        # print(f"Q: {out[available]}")
        if len(out[available]) > 1:
            logits = (out[available] - out[available].mean())
            # print(f"mlogits: {logits}")
            div = ((logits ** 2).sum() / (len(logits) - 1)) ** 0.5
            # print(f"std: {div}")

            if np.all(logits == 0.0):
                action_idx = np.random.choice(np.arange(len(logits)))
                action = np.arange(len(out))[available][action_idx]
                print("set_input", set_input)
                print("out", out)
                print("available", available)
                print("logits", logits)
                print("div", div)
                print(logits / div)

                return action, [1]

            logits = logits / div
            # print(f"logits: {logits}")

            probabilities = np.exp(beta * logits)

            norm = probabilities.sum()

            if norm == np.infty:
                action_idx = np.argmax(logits)
                action = np.arange(len(out))[available][action_idx]
                return action, 1.0

            probabilities /= norm
        else:
            probabilities = [1]
        # print(f"p: {probabilities}")
        action_idx = np.random.choice(np.arange(len(probabilities)), p=probabilities)

        action = np.arange(len(out))[available][action_idx]

        return action, np.max(probabilities)

    def soft_select_action_old(self, set_input, available, temperature):

        batched_set = set_input.unsqueeze(0)
        available_tensor = torch.FloatTensor(available)
        out = self.brain.predict(batched_set, target=False).squeeze(0)
        out = torch.FloatTensor(out)

        out = out + torch.abs(torch.min(out))
        out = out - torch.max(out * available_tensor)
        out = self.masked_softmax(out, available_tensor, dim=0, temperature=temperature)

        # print(available_tensor)
        # masked_action = action_probs #* available_tensor# + ((1 - available_tensor) * float('-inf'))
        dist = Categorical(out)
        # print(masked_action)
        action = dist.sample()
        action = action.item()
        return action, temperature

    def get_targets(self, batch):
        """
        Compute the TD-errors using the n-step Q-learning function and the model prediction
        :param batch: the batch to process
        :return: the state input, the true y, and the error for updating the memory replay
        """

        batch_len = len(batch)
        set_input, avail = list(zip(*[e[1][0] for e in batch]))

        set_batch = torch.stack(set_input)
        # states = np.array([e[1][0] for e in batch])

        next_set_input, next_avail = list(zip(*[e[1][3] for e in batch]))
        next_set_batch = torch.stack(next_set_input)


        p = self.brain.predict(set_batch, target=False)

        p_ = self.brain.predict(next_set_batch, target=False)
        p_target_ = self.brain.predict(next_set_batch, target=True)

        x = []
        y = []  # np.zeros((batch_len, self.instance_size, 1))
        errors = np.zeros(len(batch))
        weight_IS = np.zeros(len(batch))

        for i in range(batch_len):

            sample = batch[i][1]  # sample is (state_features, action, reward, next_state_features)
            state_set, state_avail = sample[0]
            action = sample[1]
            reward = sample[2]
            next_state_set, next_state_avail = sample[3]
            next_action_indices = np.argwhere(next_state_avail == 1).reshape(-1)
            t = p[i]

            q_value_prediction = t[action]

            if len(next_action_indices) == 0:
                td_q_value = reward
                t[action] = td_q_value

            else:

                mask = np.zeros(p_[i].shape, dtype=bool)
                mask[next_action_indices] = True

                best_valid_next_action_id = np.argmax(p_[i][mask])
                best_valid_next_action = np.arange(len(mask))[mask.reshape(-1)][best_valid_next_action_id]

                td_q_value = reward + GAMMA * p_target_[i][best_valid_next_action]  # double DQN
                t[action] = td_q_value

            state = (state_set, state_avail)
            x.append(state)
            y.append(t)

            errors[i] = abs(q_value_prediction - td_q_value)

            weight_IS[i] = batch[i][2]

        return x, y, errors, weight_IS,  # global_mask

    def learning(self):
        """
        execute a learning step on a batch of randomly selected experiences from the memory
        :return: the subsequent loss
        """

        batch = self.memory.sample(self.args.batch_size)

        x, y, errors, weight_IS = self.get_targets(batch)

        # update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        loss = self.brain.train(x, y, weight_IS)

        return round(loss, 4)

