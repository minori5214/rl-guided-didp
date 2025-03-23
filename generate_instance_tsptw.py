from rl_agent.hybrid_cp_rl_solver.problem.tsptw.environment.tsptw import TSPTW
import os
import numpy as np
import random

class InstanceGenerator():
    def __init__(self, n_city=20, grid_size=100, max_tw_gap=100, max_tw_size=1000, seed=0):
        self.n_city = n_city
        self.grid_size = grid_size
        self.max_tw_gap = max_tw_gap
        self.max_tw_size = max_tw_size
        self.seed = seed

    def generate(self, num_instance=1, is_integer_instance=True):
        random.seed(self.seed)
        np.random.seed(self.seed)

        save_folder = "./instances/tsptw/Cappart/n%d/gs%d-tw-%d-%d-ni%d-s%d" % \
                        (self.n_city, self.grid_size, 
                         self.max_tw_gap, self.max_tw_size,
                         num_instance, self.seed)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for i in range(num_instance):
            seed = np.random.randint(1000000)

            instance =  TSPTW.generate_random_instance(
                            n_city=self.n_city, grid_size=self.grid_size,
                            max_tw_gap=self.max_tw_gap, max_tw_size=self.max_tw_size,
                            seed=seed, is_integer_instance=is_integer_instance)

            # Write the instance information in a .txt file
            with open(os.path.join(save_folder, f'{i}.txt'), 'w') as f:
                # n_city
                f.write(str(instance.n_city) + '\n')

                # Travel time
                for i in range(instance.n_city):
                    for j in range(instance.n_city):
                        f.write(str(instance.travel_time[i][j]) + ' ')
                    f.write('\n')
                # Time windows
                for i in range(instance.n_city):
                    f.write(str(int(instance.time_windows[i][0])) + ' ' + str(int(instance.time_windows[i][1])) + '\n')

                # x_coord
                for i in range(instance.n_city):
                    f.write(str(instance.x_coord[i]) + ' ')
                f.write('\n')

                # y_coord
                for i in range(instance.n_city):
                    f.write(str(instance.y_coord[i]) + ' ')


if __name__ == "__main__":
    n_city = 100
    grid_size = 100
    max_tw_gap = 100
    max_tw_size = 1000
    seed = 0

    instance_generator = InstanceGenerator(
                            n_city=n_city, 
                            grid_size=grid_size,
                            max_tw_gap=max_tw_gap, 
                            max_tw_size=max_tw_size,
                            seed=seed
                            )
    instance_generator.generate(num_instance=100, is_integer_instance=True)