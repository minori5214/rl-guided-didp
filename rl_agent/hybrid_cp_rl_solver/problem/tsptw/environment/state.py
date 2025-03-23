

class State:
    def __init__(self, instance, must_visit, last_visited, cur_time, tour):
        """
        Build a State
        Note that the set of valid actions correspond to the must_visit part of the state
        :param instance: the problem instance considered
        :param must_visit: cities that still have to be visited.
        :param last_visited: the current location
        :param cur_time: the current time
        :param tour: the tour that is currently done
        """

        self.instance = instance
        self.must_visit = must_visit
        self.last_visited = last_visited
        self.cur_time = cur_time
        self.tour = tour
        self.use_state_constraint = False # off for now

    def step(self, action):
        """
        Performs the transition function of the DP model
        :param action: the action selected
        :return: the new state wrt the transition function on the current state T(s,a) = s'
        """

        new_must_visit = self.must_visit - set([action])
        new_last_visited = action
        new_cur_time = max(self.cur_time + self.instance.travel_time[self.last_visited][action],
                       self.instance.time_windows[action][0])
        new_tour = self.tour + [new_last_visited]

        #  Application of the validity conditions and the pruning rules before creating the new state
        new_must_visit = self.prune_invalid_actions(new_must_visit, new_last_visited, new_cur_time)
        new_must_visit = self.prune_dominated_actions(new_must_visit, new_cur_time)
        if self.use_state_constraint:
            new_must_visit = self.state_constraint(new_must_visit, new_last_visited, new_cur_time)

        new_state = State(self.instance, new_must_visit, new_last_visited, new_cur_time, new_tour)

        return new_state

    def is_done(self):
        """
        :return: True iff there is no remaining actions 
                 or there is any city where the time windows are exceeded
        """
        #print("Check", self.tour, self.must_visit)
        return len(self.must_visit) == 0

        # Early termination
        #return len(self.must_visit) == 0 or \
        #        len(self.tour) + len(self.must_visit) < self.instance.n_city

    def is_success(self):
        """
        :return: True iff there is the tour is fully completed
        """

        return len(self.tour) == self.instance.n_city

    def state_constraint(self, new_must_visit, new_last_visited, new_cur_time):
        """
        Prune customer i if visiting any customer j after i will violate the time window of j
        """

        pruned_must_visit = []

        for a in new_must_visit:
            check = [a != b and new_cur_time + \
                     self.instance.travel_time[new_last_visited][a] + \
                        self.instance.travel_time[a][b] > self.instance.time_windows[b][1] \
                            for b in new_must_visit]
            if sum(check) == 0: # no customer j visited through customer i will violate TW j
                pruned_must_visit.append(a)

        return set(pruned_must_visit)

    def prune_invalid_actions(self, new_must_visit, new_last_visited, new_cur_time):
        """
        Validity condition: Keep only the cities that can fit in the time windows according to the travel time.
        :param new_must_visit: the cities that we still have to visit
        :param new_last_visited: the city where we are
        :param new_cur_time: the current time
        :return:
        """

        pruned_must_visit = [a for a in new_must_visit if
                             new_cur_time + self.instance.travel_time[new_last_visited][a] <= self.instance.time_windows[a][1]]

        return set(pruned_must_visit)

    def prune_dominated_actions(self, new_must_visit, new_cur_time):
        """
        Pruning dominated actions: We remove all the cities having their time windows exceeded
        :param new_must_visit: the cities that we still have to visit
        :param new_cur_time: the current time
        :return:
        """

        pruned_must_visit = [a for a in new_must_visit if self.instance.time_windows[a][1] >= new_cur_time]

        return set(pruned_must_visit)
