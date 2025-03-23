import networkx as nx
import random
import heapq
import numpy as np
import torch

class TSP:

    def __init__(self, n_city, travel_time, x_coord, y_coord):
        """
        Create an instance of the TSP problem
        :param n_city: number of cities
        :param travel_time: travel time matrix between the cities
        :param x_coord: list of x-pos of the cities
        :param y_coord: list of y-pos of the cities
        """

        self.n_city = n_city
        self.travel_time = travel_time
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.graph = self.build_graph()

    def build_graph(self):
        """
        Build a networkX graph representing the TSPTW instance. Features on the edges are the distances
        and 4 binary values stating if the edge is part of the (1, 5, 10, 20) nearest neighbors of a node?
        :return: the graph
        """

        g = nx.DiGraph()

        for i in range(self.n_city):

            cur_travel_time = self.travel_time[i][:]

            # +1 because we remove the self-edge (cost 0)
            k_min_idx_1 = heapq.nsmallest(1 + 1, range(len(cur_travel_time)), cur_travel_time.__getitem__)
            k_min_idx_5 = heapq.nsmallest(5 + 1, range(len(cur_travel_time)), cur_travel_time.__getitem__)
            k_min_idx_10 = heapq.nsmallest(10 + 1, range(len(cur_travel_time)), cur_travel_time.__getitem__)
            k_min_idx_20 = heapq.nsmallest(20 + 1, range(len(cur_travel_time)), cur_travel_time.__getitem__)

            for j in range(self.n_city):

                if i != j:

                    is_k_neigh_1 = 1 if j in k_min_idx_1 else 0
                    is_k_neigh_5 = 1 if j in k_min_idx_5 else 0
                    is_k_neigh_10 = 1 if j in k_min_idx_10 else 0
                    is_k_neigh_20 = 1 if j in k_min_idx_20 else 0

                    weight = self.travel_time[i][j]
                    g.add_edge(i, j, weight=weight, is_k_neigh_1=is_k_neigh_1, is_k_neigh_5=is_k_neigh_5,
                               is_k_neigh_10=is_k_neigh_10, is_k_neigh_20=is_k_neigh_20)

        assert g.number_of_nodes() == self.n_city

        return g

    def get_edge_feat_tensor(self, max_dist):
        """
        Return a tensor of the edges features.
        As the features for the edges are not state-dependent, we can pre-compute them
        :param max_dist: the maximum_distance possible given the grid-size
        :return: a torch tensor of the features
        """

        edge_feat = [[e[2]["weight"] / max_dist,
                      e[2]["is_k_neigh_1"],
                      e[2]["is_k_neigh_5"],
                      e[2]["is_k_neigh_10"],
                      e[2]["is_k_neigh_20"]]
                     for e in self.graph.edges(data=True)]

        edge_feat_tensor = torch.FloatTensor(edge_feat)

        return edge_feat_tensor


    @staticmethod
    def generate_random_instance(n_city, grid_size,
                                 is_integer_instance, seed):
        """
        :param n_city: number of cities
        :param grid_size: x-pos/y-pos of cities will be in the range [0, grid_size]
        :param is_integer_instance: True if we want the distances and time widows to have integer values
        :param seed: seed used for generating the instance. -1 means no seed (instance is random)
        :return: a feasible TSPTW instance randomly generated using the parameters
        """

        rand = random.Random()

        if seed != -1:
            rand.seed(seed)

        x_coord = [rand.uniform(0, grid_size) for _ in range(n_city)]
        y_coord = [rand.uniform(0, grid_size) for _ in range(n_city)]

        travel_time = []

        for i in range(n_city):

            dist = [float(np.sqrt((x_coord[i] - x_coord[j]) ** 2 + (y_coord[i] - y_coord[j]) ** 2))
                    for j in range(n_city)]

            if is_integer_instance:
                dist = [round(x) for x in dist]

            travel_time.append(dist)
        
        return TSP(n_city, travel_time, x_coord, y_coord)

    @staticmethod
    def generate_dataset(size, n_city, grid_size, is_integer_instance, seed):
        """
        Generate a dataset of instance
        :param size: the size of the data set
        :param n_city: number of cities
        :param grid_size: x-pos/y-pos of cities will be in the range [0, grid_size]
        :param is_integer_instance: True if we want the distances and time widows to have integer values
        :param seed: the seed used for generating the instance
        :return: a dataset of '#size' feasible TSPTW instance randomly generated using the parameters
        """

        dataset = []
        for i in range(size):
            new_instance = TSP.generate_random_instance(n_city=n_city, grid_size=grid_size,
                                                          is_integer_instance=is_integer_instance, seed=seed)
            dataset.append(new_instance)
            seed += 1

        return dataset
