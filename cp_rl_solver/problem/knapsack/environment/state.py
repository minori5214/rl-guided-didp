
class State:
    def __init__(self, instance, weight, stage, available_action):

        self.instance = instance
        self.weight = weight
        self.stage = stage
        self.available_action = available_action


    def step(self, action):

        collected_profit = action * self.instance.profit_list[self.stage]
        new_weight = self.weight + action * self.instance.weight_list[self.stage]

        if self.stage + 1 == self.instance.n_item:
            new_available_action = set()
        elif new_weight + self.instance.weight_list[self.stage + 1] > self.instance.capacity:
            new_available_action = set([0])
        else:
            new_available_action = set([0, 1])

        new_stage = self.stage + 1
        new_state = State(self.instance, new_weight, new_stage, new_available_action)

        assert new_weight <= self.instance.capacity
        return new_state, collected_profit

    def is_done(self):

        return self.stage == self.instance.n_item

