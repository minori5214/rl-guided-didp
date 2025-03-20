import random
import networkx as nx

class Knapsack:

    def __init__(self, n_item, capacity, weight_list, profit_list, sort=True):
        self.n_item = n_item
        if sort:
            ratio_list = [profit_list[i]/weight_list[i] for i in range(n_item)]
            sorted_weight_profit = sorted(zip(ratio_list, weight_list, profit_list), reverse=True)
            self.weight_list = [w for _, w, _ in sorted_weight_profit]
            self.profit_list = [p for _, _, p in sorted_weight_profit]
        else:
            self.weight_list = weight_list
            self.profit_list = profit_list

        self.index_list = list(range(0, n_item))
        self.object_list = list(zip(self.index_list, self.weight_list, self.profit_list))
        self.capacity = capacity
        self.graph = self.build_graph()

    def build_graph(self):

        g = nx.DiGraph()

        for i in range(self.n_item):

            for j in range(self.n_item):

                if i != j:

                    g.add_edge(i, j)

        assert g.number_of_nodes() == self.n_item

        return g

    def __repr__(self):
        return "nItem: %d ; size: %d ; objects: %s" % \
               (self.n_item, self.capacity, self.object_list)

    # Adapted from http://www.dcs.gla.ac.uk/~pat/cpM/jchoco/knapsack/papers/hardInstances.pdf


    @staticmethod
    def generate_random_instance(n_item, lb, ub, ratio, cor_type, seed=-1, is_integer_instance=False, sort=True):

        rand = random.Random()

        if seed != -1:
            rand.seed(seed)



        if cor_type == "uncorrelated":
            weight_list = [rand.uniform(lb, ub) for _ in range(n_item)]
            profit_list = [rand.uniform(lb, ub) for _ in range(n_item)]

        elif cor_type == "weakly":
            weight_list = [rand.uniform(lb, ub) for _ in range(n_item)]
            profit_list = [max(1.0, rand.uniform(weight_list[i] - (ub / 10.), weight_list[i] + (ub / 10.))) for i in
                           range(n_item)]
        elif cor_type == "strongly":
            weight_list = [rand.uniform(lb, ub) for _ in range(n_item)]
            profit_list = [weight_list[i] + ub / 10. for i in range(n_item)]

        elif cor_type == "spanned":
            subset_size = 2
            multiplier = 10.

            subset_weight = [rand.uniform(lb, ub) for _ in range(subset_size)]
            subset_profit = [subset_weight[i] + ub / 10. for i in range(subset_size)]

            subset_weight = [2. * x / multiplier for x in subset_weight]
            subset_profit = [2. * x / multiplier for x in subset_profit]

            weight_list = []
            profit_list = []

            for i in range(n_item):

                factor = rand.uniform(1, multiplier)
                item_idx = rand.randint(0, subset_size - 1)

                weight = factor * subset_weight[item_idx]
                weight_list.append(weight)

                profit = factor * subset_profit[item_idx]
                profit_list.append(profit)

        else:
            raise Exception("Correlation type not recognized")

        if is_integer_instance:
            weight_list = [int(x) for x in weight_list]
            profit_list = [int(x) for x in profit_list]

        capacity = int(ratio * sum(weight_list))

        return Knapsack(n_item, capacity, weight_list, profit_list, sort=sort)

    @staticmethod
    def generate_dataset(size, n_item, lb, ub, capacity_ratio, cor_type, seed, is_integer_instance=False, sort=True):
        """
        Generate a dataset of instance
        :param size: the size of the data set
        :param n_item: number of items
        :param lb: lower bound for the weight and profit of the items
        :param ub: upper bound for the weight and profit of the items
        :param capacity_ratio: the capacity of the knapsack is capacity_ratio * sum(weight_list)
        :param cor_type: the correlation type of the instance
        :param seed: the seed used for generating the instance
        :param is_integer_instance: True if we want the distances and time widows to have integer values
        :return: a dataset of '#size' feasible TSPTW instance randomly generated using the parameters
        """

        dataset = []
        for i in range(size):
            new_instance = Knapsack.generate_random_instance(n_item=n_item, lb=lb, ub=ub,
                                                             ratio=capacity_ratio, cor_type=cor_type,
                                                             is_integer_instance=is_integer_instance,
                                                             seed=seed, sort=sort)
            dataset.append(new_instance)
            seed += 1

        return dataset