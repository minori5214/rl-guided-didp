
import torch
import torch.optim as optim
import torch.nn.functional as F

import os
import numpy as np
import dgl

from rl_agent.hybrid_cp_rl_solver.architecture.graph_attention_network import GATNetwork
from rl_agent.hybrid_cp_rl_solver.architecture.graph_attention_network_onnx import GATNetworkONNX


class BrainDQN:
    """
    Definition of the DQN Brain, computing the DQN loss
    """

    def __init__(self, args, num_node_feat, num_edge_feat):
        """
        Initialization of the DQN Brain
        :param args: argparse object taking hyperparameters
        :param num_node_feat: number of features on the nodes
        :param num_edge_feat: number of features on the edges
        """

        self.args = args

        self.embedding = [(num_node_feat, num_edge_feat),
                         (self.args.latent_dim, self.args.latent_dim),
                         (self.args.latent_dim, self.args.latent_dim),
                         (self.args.latent_dim, self.args.latent_dim)]

        if args.onnx:
            self.model = GATNetworkONNX(
                layer_features=self.embedding,
                n_hidden_layer=self.args.hidden_layer,
                latent_dim=self.args.latent_dim,
                output_dim=1,
                graph_pooling=False
            )
            self.target_model = GATNetworkONNX(
                layer_features=self.embedding,
                n_hidden_layer=self.args.hidden_layer,
                latent_dim=self.args.latent_dim,
                output_dim=1,
                graph_pooling=False
            )
        else:
            self.model = GATNetwork(self.embedding, self.args.hidden_layer, self.args.latent_dim, 1)
            self.target_model = GATNetwork(self.embedding, self.args.hidden_layer, self.args.latent_dim, 1)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        if self.args.mode == 'gpu':
            self.model.cuda()
            self.target_model.cuda()

    def train(self, x, y):
        """
        Compute the loss between (f(x) and y)
        :param x: the input
        :param y: the true value of y
        :return: the loss
        """

        self.model.train()

        graph, _ = list(zip(*x))
        graph_batch = dgl.batch(graph)

        y_pred = self.model(graph_batch, graph_pooling=False)
        y_pred = torch.stack([g.ndata["n_feat"] for g in dgl.unbatch(y_pred)]).squeeze(dim=2)
        y_tensor = torch.FloatTensor(np.array(y))

        if self.args.mode == 'gpu':
            y_tensor = y_tensor.contiguous().cuda()

        loss = F.smooth_l1_loss(y_pred, y_tensor)
        #loss *= y_pred.shape[1] # Issue fixed; multiply by the batch size to make it back to the original scale
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_onnx(self, x, y):
        self.model.train()

        graph, _ = list(zip(*x))

        # Extract batched node features, edge_index, edge_attr
        node_features_list = []
        edge_index_list = []
        edge_attr_list = []
        batch_mapping = []

        node_offset = 0
        for batch_id, g in enumerate(graph):
            x = g.ndata["n_feat"]
            edge_attr = g.edata["e_feat"]
            src, dst = g.edges()
            edge_index = torch.stack([src, dst], dim=0) + node_offset

            node_features_list.append(x)
            edge_index_list.append(edge_index)
            edge_attr_list.append(edge_attr)

            batch_mapping.extend([batch_id] * x.shape[0])
            node_offset += x.shape[0]

        # Stack to get batched inputs
        x_tensor = torch.cat(node_features_list, dim=0)               # [total_nodes, node_feat_dim]
        edge_index_tensor = torch.cat(edge_index_list, dim=1)         # [2, total_edges]
        edge_attr_tensor = torch.cat(edge_attr_list, dim=0)           # [total_edges, edge_feat_dim]

        # Move to GPU if needed
        if self.args.mode == 'gpu':
            x_tensor = x_tensor.cuda()
            edge_index_tensor = edge_index_tensor.cuda()
            edge_attr_tensor = edge_attr_tensor.cuda()

        y_pred = self.model(x_tensor, edge_index_tensor, edge_attr_tensor)
        y_pred = y_pred.view(-1)
        y_tensor = torch.FloatTensor(np.array(y)).view(-1).to(x_tensor.device)

        loss = F.smooth_l1_loss(y_pred.squeeze(), y_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, graph, target):
        """
        Predict the Q-values using the current graph, either using the model or the target model
        :param graph: the graph serving as input
        :param target: True is the target network must be used for the prediction
        :return: A list of the predictions for each node
        """

        with torch.no_grad():
            if target:
                self.target_model.eval()
                res = self.target_model(graph, graph_pooling=False)
            else:
                self.model.eval()
                res = self.model(graph, graph_pooling=False)

        res = dgl.unbatch(res)
        return [r.ndata["n_feat"].data.cpu().numpy().flatten() for r in res]

    def predict_onnx(self, graph, target):
        # Extract batched node features, edge_index, edge_attr
        node_features_list = []
        edge_index_list = []
        edge_attr_list = []
        batch_mapping = []
        node_counts = []

        node_offset = 0
        for batch_id, g in enumerate(dgl.unbatch(graph)):
            x = g.ndata["n_feat"]
            edge_attr = g.edata["e_feat"]
            src, dst = g.edges()
            edge_index = torch.stack([src, dst], dim=0) + node_offset

            node_features_list.append(x)
            edge_index_list.append(edge_index)
            edge_attr_list.append(edge_attr)

            batch_mapping.extend([batch_id] * x.shape[0])
            node_counts.append(x.shape[0])
            node_offset += x.shape[0]

        # Stack to get batched inputs
        x_tensor = torch.cat(node_features_list, dim=0)               # [total_nodes, node_feat_dim]
        edge_index_tensor = torch.cat(edge_index_list, dim=1)         # [2, total_edges]
        edge_attr_tensor = torch.cat(edge_attr_list, dim=0)           # [total_edges, edge_feat_dim]

        # Move to GPU if needed
        if self.args.mode == 'gpu':
            x_tensor = x_tensor.cuda()
            edge_index_tensor = edge_index_tensor.cuda()
            edge_attr_tensor = edge_attr_tensor.cuda()
        
        #print("chchcc", x_tensor.shape, edge_index_tensor.shape, edge_attr_tensor.shape)

        if target:
            res = self.target_model(x_tensor, edge_index_tensor, edge_attr_tensor)
        else:
            res = self.model(x_tensor, edge_index_tensor, edge_attr_tensor)
        
        res = res.detach().cpu().numpy()
        #print("res", res.shape)

        # Split the result into list of arrays per graph
        split_res = np.split(res, np.cumsum(node_counts)[:-1])
        #print("split res", type(split_res), split_res[0].shape)

        # Optionally flatten each per-graph result if needed
        return split_res

        #return res.detach().cpu().numpy().flatten()

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
