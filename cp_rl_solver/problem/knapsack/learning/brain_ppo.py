
import random
import os

import torch
import torch.nn as nn

from src.problem.knapsack.learning.actor_critic import ActorCritic

class BrainPPO:
    def __init__(self, args, num_node_feat):

        self.args = args

        self.policy = ActorCritic(self.args, num_node_feat)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.learning_rate)
        self.policy_old = ActorCritic(self.args, num_node_feat)
        self.policy_old.load_state_dict(self.policy.state_dict())

        if args.mode == 'gpu':
            self.policy.cuda()
            self.policy_old.cuda()

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        acc_reward = 0

        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                acc_reward = 0
            acc_reward = reward + acc_reward
            rewards.insert(0, acc_reward)
        # Optimize policy for K epochs:

        for k in range(self.args.k_epochs):

            mem = list(zip(memory.actions, memory.availables, memory.states, memory.logprobs, rewards))
            random.shuffle(mem)
            mem_actions, mem_availables, mem_states, mem_logprobs, mem_rewards = zip(*mem)

            n_batch = self.args.update_timestep // self.args.batch_size

            for j in range(n_batch):

                start_idx = j * self.args.batch_size
                end_idx = (j + 1) * self.args.batch_size - 1

                old_states_for_action = torch.stack(mem_states[start_idx:end_idx])
                old_states_for_value = torch.stack(mem_states[start_idx:end_idx])
                old_actions = torch.stack(mem_actions[start_idx:end_idx])
                old_logprobs = torch.stack(mem_logprobs[start_idx:end_idx])
                old_availables = torch.stack(mem_availables[start_idx:end_idx])
                rewards_tensor = torch.tensor(mem_rewards[start_idx:end_idx])

                # on GPU
                if self.args.mode == 'gpu':
                    old_actions = old_actions.cuda()
                    old_logprobs = old_logprobs.cuda()
                    old_availables = old_availables.cuda()
                    rewards_tensor = rewards_tensor.cuda()

                # Normalizing the rewards:

                # Evaluating old actions and values :
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_for_action, old_states_for_value,
                                                                            old_actions, old_availables)

                # Finding the ratio (pi_theta / pi_theta__old):

                ratios = torch.exp(logprobs - old_logprobs.detach())

                advantages = rewards_tensor - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards_tensor) - self.args.entropy_value * dist_entropy

                self.optimizer.zero_grad()

                loss.mean().backward()

                self.optimizer.step()


        #print("[LOG] entropy:", dist_entropy.cpu().mean().item())
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)

        torch.save(self.policy_old.state_dict(), filepath)
