import torch
import torch.nn.functional as F
import torch.optim as optim

import os
import numpy as np

from src.architecture.set_transformer import SetTransformer
from src.problem.knapsack.learning.knapsack_neural_network import KnapsackNetworkTanh

class BrainDQN:
    """
    Definition of the DQN Brain, computing the DQN loss
    """

    def __init__(self, args, num_node_feats, num_edge_feats):
        """
        Initialization of the DQN Brain
        :param args: argparse object taking hyperparameters
        :param num_node_feat: number of features on the nodes
        :param num_edge_feat: number of features on the edges
        """

        self.args = args

        if args.gnn == 'settransformer':
            self.model = SetTransformer(dim_hidden=args.latent_dim, dim_input=num_node_feats, dim_output=2)
            self.target_model = SetTransformer(dim_hidden=args.latent_dim, dim_input=num_node_feats, dim_output=2)
        elif args.gnn == 'knapsacktanh':
            self.model = KnapsackNetworkTanh(args.latent_dim, args.hidden_layer, x_dim=num_node_feats, pool='mean')
            self.target_model = KnapsackNetworkTanh(args.latent_dim, args.hidden_layer, x_dim=num_node_feats, pool='mean')

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        #NOTE: self.optimizer = RAdam(self.model.parameters(), lr=INITIAL_LR, weight_decay=self.args.L2_penalty)

        if self.args.mode == 'gpu':
            self.model.cuda()
            self.target_model.cuda()

    def decay_lr_by(self, factor):
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] * factor

        return g['lr']

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

        return g['lr']

    def weighted_mse_loss(self, y_pred, y_true, weight):

        loss_vector = ((y_pred - y_true) ** 2) * weight

        loss = loss_vector.sum() / y_pred.data.nelement()
        return loss

    def weighted_huber_loss(self, y_pred, y_true, weight, huber_delta=1):

        err = torch.abs(y_pred - y_true)

        cond = (err < huber_delta).float()

        l2 = (1 / 2) * err ** 2
        l1 = huber_delta * (err - (1 / 2) * huber_delta)

        loss = cond * l2 + (1 - cond) * l1

        loss = weight * loss.mean(dim=-1)

        return loss.mean()

    def train(self, x, y, weight_IS):
        """
        Compute the loss between (f(x) and y)
        :param x: the input
        :param y: the true value of y
        :return: the loss
        """

        self.model.train()
        set_input, valid_actions = list(zip(*x))
        batched_set = torch.stack(set_input)

        y_pred = self.model(batched_set)
        y_tensor = torch.FloatTensor(np.array(y))

        weight_tensor = torch.FloatTensor(weight_IS).unsqueeze(1)

        if self.args.mode == 'gpu':
            y_tensor = y_tensor.contiguous().cuda()
            weight_tensor = weight_tensor.contiguous().cuda()

        loss = F.smooth_l1_loss(y_pred, y_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, nn_input, target):
        """
        Predict the Q-values using the current state, either using the model or the target model
        :param nn_input: the featurized state
        :param target: True is the target network must be used for the prediction
        :return: A list of the predictions for each node
        """

        with torch.no_grad():

            if target:
                self.target_model.eval()
                res = self.target_model(nn_input)
            else:
                self.model.eval()
                res = self.model(nn_input)

        return res.cpu().numpy()

    def update_target_model(self):
        """
        Synchronise the target network with the current one
        """

        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, folder, filename):
        """
        Save the model
        :param folder: Folder requested
        :param filename: file name requested
        """

        filepath = os.path.join(folder, filename)

        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save(self.model.state_dict(), filepath)