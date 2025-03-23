from rl_agent.hybrid_cp_rl_solver.problem.portfolio.environment.portfolio import Portfolio
import os
import numpy as np
import random

class InstanceGenerator():
    def __init__(self, n_item=20, lb=0, ub=100, capacity_ratio=0.5, 
                 moment_factors=[1, 5, 5, 5], seed=0):
        self.n_item = n_item
        self.lb = lb
        self.ub = ub
        self.capacity_ratio = capacity_ratio
        self.moment_factors = moment_factors
        self.seed = seed

    def generate(self, num_instance=1, is_integer_instance=True):
        random.seed(self.seed)
        np.random.seed(self.seed)

        save_folder = "./instances/portfolio/n%d/lb%d-ub%d-cr%s-lmd%d%d%d%d-seed%d" % \
                        (self.n_item, self.lb, self.ub,                   
                         str(self.capacity_ratio).replace('.', ''),
                         self.moment_factors[0], self.moment_factors[1], 
                         self.moment_factors[2], self.moment_factors[3],
                         self.seed)
                        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for i in range(num_instance):
            seed = np.random.randint(1000000)

            instance =  Portfolio.generate_random_instance(
                            n_item=self.n_item, lb=self.lb, ub=self.ub, 
                            capacity_ratio=self.capacity_ratio, 
                            moment_factors=self.moment_factors, 
                            is_integer_instance=is_integer_instance, 
                            seed=seed)

            # Write the instance information in a .txt file
            with open(os.path.join(save_folder, f'{i}.txt'), 'w') as f:
                # n_city
                f.write(str(instance.n_item) + '\n')

                # capacity
                f.write(str(instance.capacity) + '\n')

                # weights
                for i in range(instance.n_item):
                    f.write(str(instance.weights[i]) + ' ')
                f.write('\n')

                # means
                for i in range(instance.n_item):
                    f.write(str(instance.means[i]) + ' ')
                f.write('\n')

                # deviations
                for i in range(instance.n_item):
                    f.write(str(instance.deviations[i]) + ' ')
                f.write('\n')

                # skewnesses
                for i in range(instance.n_item):
                    f.write(str(instance.skewnesses[i]) + ' ')
                f.write('\n')

                # kurtosis
                for i in range(instance.n_item):
                    f.write(str(instance.kurtosis[i]) + ' ')
                f.write('\n')

                # moment_factors
                for i in range(4):
                    f.write(str(instance.moment_factors[i]) + ' ')
                f.write('\n')



if __name__ == "__main__":
    n_items = [20, 50]
    lb = 0
    ub = 100
    capacity_ratio = 0.5
    moment_factors = [1, 5, 5, 5]
    seed = 0

    for n_item in n_items:
        instance_generator = InstanceGenerator(n_item=n_item, lb=lb, ub=ub,
                                                capacity_ratio=capacity_ratio,
                                                moment_factors=moment_factors,
                                                seed=seed)
        instance_generator.generate(num_instance=100, is_integer_instance=True)
