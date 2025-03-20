

class State:
    def __init__(self, instance, must_visit, last_visited, tour):
        """
        Build a State
        Note that the set of valid actions correspond to the must_visit part of the state
        :param instance: the problem instance considered
        :param must_visit: cities that still have to be visited.
        :param last_visited: the current location
        :param tour: the tour that is currently done
        """

        self.instance = instance
        self.must_visit = must_visit
        self.last_visited = last_visited
        self.tour = tour

    def step(self, action):
        """
        Performs the transition function of the DP model
        :param action: the action selected
        :return: the new state wrt the transition function on the current state T(s,a) = s'
        """

        new_must_visit = self.must_visit - set([action])
        new_last_visited = action
        new_tour = self.tour + [new_last_visited]

        #  Application of the validity conditions and the pruning rules before creating the new state
        #new_must_visit = self.prune_invalid_actions(new_must_visit)
        #new_must_visit = self.prune_dominated_actions(new_must_visit)

        new_state = State(self.instance, new_must_visit, new_last_visited, new_tour)

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

    def prune_invalid_actions(self, new_must_visit):
        return set(new_must_visit)

    def prune_dominated_actions(self, new_must_visit):
        return set(new_must_visit)
