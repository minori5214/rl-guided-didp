import torch
import torch.nn as nn
from torch.distributions import Categorical

from cp_rl_solver.problem.knapsack.learning.knapsack_neural_network import KnapsackActionNetwork, KnapsackStateNetwork

class ActorCritic(nn.Module):
    def __init__(self, args, num_node_feats):

        super(ActorCritic, self).__init__()

        self.args = args

        # actor
        #self.action_layer = SetTransformer(dim_hidden=args.latent_dim, dim_input=num_node_feats, dim_output=2)
        self.action_layer = KnapsackActionNetwork(args.latent_dim, args.hidden_layer, x_dim=num_node_feats, pool='mean')

        # critic
        #self.value_layer = SetTransformer(dim_hidden=args.latent_dim, dim_input=num_node_feats, dim_output=1)#
        self.value_layer = KnapsackStateNetwork(args.latent_dim, args.hidden_layer, x_dim=num_node_feats, pool='mean')


    def forward(self):
        raise NotImplementedError

    def act(self, nn_input, available_tensor):

        if self.args.mode == "gpu":
            available_tensor = available_tensor.cuda()
        batched_nn_input = nn_input.unsqueeze(0)

        self.action_layer.eval()
        with torch.no_grad():
            out = self.action_layer(batched_nn_input)

            action_probs = out.squeeze(0)
            action_probs = action_probs + torch.abs(torch.min(action_probs))
            action_probs = action_probs - torch.max(action_probs * available_tensor)

            action_probs = self.masked_softmax2(action_probs, available_tensor, dim=0)

            dist = Categorical(action_probs)
            action = dist.sample()

        return action, dist.log_prob(action), action_probs

    def evaluate(self, state_for_action, state_for_value, action, available_tensor):

        if self.args.mode == "gpu":
            available_tensor = available_tensor.cuda()

        action_probs = self.action_layer(state_for_action)

        action_probs = action_probs + torch.abs(torch.min(action_probs, 1, keepdim=True)[0]) # positive range
        action_probs = action_probs - torch.max(action_probs * available_tensor, 1, keepdim=True)[0]

        action_probs = self.masked_softmax2(action_probs, available_tensor, dim=1)

        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)

        dist_entropy = dist.entropy()
        state_value = self.value_layer(state_for_value)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def masked_softmax(self, vec, mask, dim=1, epsilon=1e-5):

        if self.args.mode == "gpu":
            mask = mask.cuda()

        exps = torch.exp(vec)
        masked_exps = exps * mask.float()

        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
        return (masked_exps / masked_sums)

    def masked_softmax2(self, vector: torch.Tensor, mask: torch.Tensor, dim: int = -1,
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