#!/usr/bin/env python3

import argparse
import copy
import time
import os
import math
import numpy as np

import didppy as dp
import read_tsptw

import torch
torch.set_num_threads(1)
print("cpu thread used: ", torch.get_num_threads())
os.environ['MKL_NUM_THREADS'] = '1'

#start = time.perf_counter()

def get_max_tw_value(n_city, max_tw_size, max_tw_gap):
    #    :param max_tw_gap: maximum time windows gap allowed between the cities (used for normalization purpose)
    #    :param max_tw_size: time windows of cities will be in the range [0, max_tw_size] (used for normalization purpose)
    return (n_city - 1) * (max_tw_size + max_tw_gap)

def create_heuristic_function(unvisited, location, time, x_coord, y_coord, nn):
    def evaluate(state):
        u = state[unvisited]
        loc = state[location]
        t = state[time]
        x = x_coord[loc]
        y = y_coord[loc]

        return nn.eval(u, loc, t, x, y)
    
    return evaluate


def create_model_kuroiwa(n, nodes, edges, a, b, non_zero_base_case, target=None):
    if target is None:
        target_unvisited = [i for i in range(1, n)]
        target_location = 0
        target_time = 0
    else:
        assert type(target) == list
        assert len(target) == 3
        assert type(target[0]) == list
        assert type(target[1]) == int
        assert type(target[2]) == int
        target_unvisited = target[0]
        target_location = target[1]
        target_time = target[2]

    model = dp.Model()

    customer = model.add_object_type(number=n)
    unvisited = model.add_set_var(object_type=customer, target=target_unvisited)
    location = model.add_element_var(object_type=customer, target=target_location)
    time = model.add_int_resource_var(target=target_time, less_is_better=True)

    distance_matrix = [
        [edges[i, j] if (i, j) in edges else 0 for j in nodes] for i in nodes
    ]
    distance = model.add_int_table(distance_matrix)

    shortest_distance_matrix = copy.deepcopy(distance_matrix)

    for k in range(1, n):
        for i in range(n):
            for j in range(n):
                d = shortest_distance_matrix[i][k] + shortest_distance_matrix[k][j]

                if shortest_distance_matrix[i][j] > d:
                    shortest_distance_matrix[i][j] = d

    shortest_distance = model.add_int_table(shortest_distance_matrix)

    for i in range(1, n):
        model.add_state_constr(
            ~(unvisited.contains(i)) | (time + shortest_distance[location, i] <= b[i])
        )

    if non_zero_base_case:
        model.add_base_case([unvisited.is_empty()], cost=distance[location, 0])
    else:
        model.add_base_case([location == 0, unvisited.is_empty()])

    state_cost = dp.IntExpr.state_cost()
    name_to_customer = {}

    for i in range(1, n):
        name = "visit {}".format(i)
        name_to_customer[name] = i
        visit = dp.Transition(
            name=name,
            cost=distance[location, i] + state_cost,
            effects=[
                (unvisited, unvisited.remove(i)),
                (location, i),
                (time, dp.max(time + distance[location, i], a[i])),
            ],
            preconditions=[unvisited.contains(i), time + distance[location, i] <= b[i]],
        )
        model.add_transition(visit)

    if not non_zero_base_case:
        name = "return"
        name_to_customer[name] = 0
        return_to_depot = dp.Transition(
            name=name,
            cost=distance[location, 0] + state_cost,
            effects=[(location, 0), (time, time + distance[location, 0])],
            preconditions=[unvisited.is_empty(), location != 0],
        )
        model.add_transition(return_to_depot)

    min_distance_to = model.add_int_table(
        [min(distance_matrix[i][j] for i in nodes if i != j) for j in nodes]
    )

    if non_zero_base_case:
        model.add_dual_bound(min_distance_to[unvisited] + min_distance_to[0])
    else:
        model.add_dual_bound(
            min_distance_to[unvisited]
            + (location != 0).if_then_else(min_distance_to[0], 0)
        )

    min_distance_from = model.add_int_table(
        [min(distance_matrix[i][j] for j in nodes if i != j) for i in nodes]
    )

    if non_zero_base_case:
        model.add_dual_bound(min_distance_from[unvisited] + min_distance_from[location])
    else:
        model.add_dual_bound(
            min_distance_from[unvisited]
            + (location != 0).if_then_else(min_distance_from[location], 0)
        )

    return model, name_to_customer, unvisited, location, time


def create_model_cappart(n, nodes, edges, a, b):
    model = dp.Model()

    customer = model.add_object_type(number=n)
    unvisited = model.add_set_var(object_type=customer, target=[i for i in range(1, n)])
    location = model.add_element_var(object_type=customer, target=0)
    time = model.add_int_var(target=0)

    distance_matrix = [
        [edges[i, j] if (i, j) in edges else 0 for j in nodes] for i in nodes
    ]
    distance = model.add_int_table(distance_matrix)

    for i in range(1, n):
        model.add_state_constr(
            ~(unvisited.contains(i)) | (time < b[i])
        )

    model.add_base_case([unvisited.is_empty()], cost=distance[location, 0])

    state_cost = dp.IntExpr.state_cost()
    name_to_customer = {}

    for i in range(1, n):
        name = "visit {}".format(i)
        name_to_customer[name] = i
        visit = dp.Transition(
            name=name,
            cost=distance[location, i] + state_cost,
            effects=[
                (unvisited, unvisited.remove(i)),
                (location, i),
                (time, dp.max(time + distance[location, i], a[i])),
            ],
            preconditions=[unvisited.contains(i), time + distance[location, i] <= b[i]],
        )
        model.add_transition(visit)

    return model, name_to_customer, unvisited, location, time

def create_RL_g_function(location, edges, destination, ub_cost, 
                         scaling_factor, use_tsp_nn_model):
    def g_function(parent_g, state):
        cost = edges[state[location], destination]

        if use_tsp_nn_model:
            reward = (-cost) * scaling_factor
            g = parent_g + (-reward)
        
        else:
            # Add the negative cost and scale it
            reward = (ub_cost + (-cost)) * scaling_factor

            # Accumulate negative reward to make it minimization
            g = parent_g + (-reward)

        #print("state", state[unvisited], state[location], destination, ub_cost, "g={}".format(g))

        return g

    return g_function

def create_policy(unvisited, customers, non_zero_base_case=False):
    def policy(state):
        unvisited_set = state[unvisited]

        if not non_zero_base_case:
            unvisited_set.add(0)

        p = 1 / len(unvisited_set)

        # The order must match the order in which **non-forced** transitions are defined
        # Thus, the probability for the depot comes at the end.
        # For a forced transition, we do not need to call the policy,
        # since the probability should be 1.0
        # The probabilities of unapplicable transitions do not matter,
        # so they can be 'nan'
        probabilities = np.array(
            [
                p if j in unvisited_set else np.float64("nan")
                for j in customers[1:] + [0]
            ],
            dtype=np.float64,
        )

        # Policy must return log probabilities
        log_probabilities = np.log(probabilities)

        return log_probabilities

    return policy

def create_RL_policy(unvisited, customers, location, time_var, grid_size, max_tw_value, 
                        model_folder, input, train_n_city, train_grid_size, 
                        train_max_tw_gap, train_max_tw_size, 
                        softmax_temperature,
                        seed, rl_algorithm, non_zero_base_case=False,
                        time_window_start=None, time_window_end=None,
                        use_tsp_nn_model=False):

    import dgl

    if use_tsp_nn_model:
        from tsp_agent import TSPAgent
        from cp_rl_solver.problem.tsp.environment.environment import Environment
        from cp_rl_solver.problem.tsp.environment.state import State

        # DQN agent
        agent = TSPAgent(model_folder, input, train_n_city, train_grid_size, 
                        seed, rl_algorithm)

        # TSPTW environment
        env = Environment(agent.instance, agent.n_node_feat, agent.n_edge_feat,
                            1, grid_size)
    else:
        from tsptw_agent import TSPTWAgent
        from cp_rl_solver.problem.tsptw.environment.environment import Environment
        from cp_rl_solver.problem.tsptw.environment.state import State

        # PPO agent
        agent = TSPTWAgent(model_folder, input, train_n_city, train_grid_size, 
                        train_max_tw_gap, train_max_tw_size, seed, rl_algorithm)

        # TSPTW environment
        env = Environment(agent.instance, agent.n_node_feat, agent.n_edge_feat,
                            1, grid_size, max_tw_value)

    def to_RLstate_tsp(DIDPstate):
        must_visit = DIDPstate[unvisited]
        last_visited = DIDPstate[location]
        #print("state", must_visit, last_visited, cur_time)

        # State.tour can be left empty
        return State(agent.instance, must_visit, last_visited, tour=[])

    def to_RLstate_tsptw(DIDPstate):
        must_visit = DIDPstate[unvisited]
        last_visited = DIDPstate[location]
        cur_time = DIDPstate[time_var]
        #print("policy state", must_visit, last_visited, cur_time, time_window_start[last_visited], time_window_end[last_visited])

        # State.tour can be left empty
        return State(agent.instance, must_visit, last_visited, cur_time, tour=[])

    def policy(state):
        #print("state", state[unvisited], state[location], state[time_var])
        unvisited_set = state[unvisited]

        depot_prob = [np.float64("nan")] 
        if not non_zero_base_case and len(unvisited_set) == 0:
            depot_prob = [1.0] # depot is a forced transition, so must come at the end

        # Convert state to RL state
        if use_tsp_nn_model:
            RLstate = to_RLstate_tsp(state)
        else:
            RLstate = to_RLstate_tsptw(state)

        graph = env.make_nn_input(RLstate, 'cpu')
        avail = env.get_valid_actions(RLstate)

        available_tensor = torch.FloatTensor(avail)

        bgraph = dgl.batch([graph, ])

        res = agent.model(bgraph, graph_pooling=False)

        out = dgl.unbatch(res)[0]

        action_probs = out.ndata["n_feat"].squeeze(-1)
        #print("action prob 1", action_probs)
        #print("available_tensor", available_tensor)

        action_probs = action_probs + torch.abs(torch.min(action_probs))
        action_probs = action_probs - torch.max(action_probs * available_tensor)
        #print("action prob 2", action_probs)

        action_probs = agent.actor_critic_network.masked_softmax(
                            action_probs, available_tensor, dim=0,
                            temperature=softmax_temperature)

        #print("action prob 3", action_probs)
        #print(available_tensor)
        # If the action is available, and the probability is 0.0, we set it to the minimum possible float
        action_probs = action_probs + (1e-32 * (available_tensor == 1.0))

        action_probs = action_probs.detach()
        #print("action prob 4", action_probs)

        #p = 1 / len(unvisited_set)

        # The order must match the order in which **non-forced** transitions are defined
        # Thus, the probability for the depot comes at the end.
        # For a forced transition, we do not need to call the policy,
        # since the probability should be 1.0
        # The probabilities of unapplicable transitions do not matter,
        # so they can be 'nan'
        probabilities = np.array(
            [
                action_probs[j] if j in unvisited_set else np.float64("nan")
                for j in customers[1:]
            ] + depot_prob,
            dtype=np.float64,
        )
        #print("policy values: {}".format(probabilities))

        # Policy must return log probabilities
        log_probabilities = np.log(probabilities)
        #print("probability", probabilities)
        #print("log_probabilities", log_probabilities)
        #print("")

        return log_probabilities

    return policy

def policy_f_evaluator(log_pi, g, h):
    if h is None:
        h = 0.0 # if there is no dual bound, only g is used

    # PHS priority function in logarithm
    #print("g", g)
    #print("h", h)
    #print("log_pi", log_pi)
    f = np.log(g + h) - log_pi
    #print("f", f)
    return f

def policy_f_evaluator_for_negative(log_pi, g, h):
    # If g- and h-values are be negative, we take the negative to make it positive.
    # Then, it becomes maximization, so we multiply pi (or add log pi to the logarithm)
    # and take the negative to put it back to minimization
    f = -(np.log(-(g + h)) + log_pi)
    return f

def create_RL_heuristic_function(
    unvisited, location, time_var, grid_size, max_tw_value, 
    model_folder, input, train_n_city, train_grid_size, train_max_tw_gap, train_max_tw_size, 
    seed, rl_algorithm, use_tsp_nn_model
):
    import dgl

    if use_tsp_nn_model:
        from tsp_agent import TSPAgent
        from cp_rl_solver.problem.tsp.environment.environment import Environment
        from cp_rl_solver.problem.tsp.environment.state import State

        # DQN agent
        agent = TSPAgent(model_folder, input, train_n_city, train_grid_size, 
                        seed, rl_algorithm)

        # TSPTW environment
        env = Environment(agent.instance, agent.n_node_feat, agent.n_edge_feat,
                            1, grid_size)

    else:
        from tsptw_agent import TSPTWAgent
        from cp_rl_solver.problem.tsptw.environment.environment import Environment
        from cp_rl_solver.problem.tsptw.environment.state import State

        # DQN agent
        agent = TSPTWAgent(model_folder, input, train_n_city, train_grid_size, 
                        train_max_tw_gap, train_max_tw_size, seed, rl_algorithm)

        # TSPTW environment
        env = Environment(agent.instance, agent.n_node_feat, agent.n_edge_feat,
                            1, grid_size, max_tw_value)

    def to_RLstate_tsp(DIDPstate):
        must_visit = DIDPstate[unvisited]
        last_visited = DIDPstate[location]
        #print("state", must_visit, last_visited)

        # State.tour can be left empty
        return State(agent.instance, must_visit, last_visited, tour=[])

    def to_RLstate_tsptw(DIDPstate):
        must_visit = DIDPstate[unvisited]
        last_visited = DIDPstate[location]
        cur_time = DIDPstate[time_var]
        #print("state", must_visit, last_visited, cur_time)

        # State.tour can be left empty
        return State(agent.instance, must_visit, last_visited, cur_time, tour=[])

    def RL_heuristic(state):
        # Convert state to RL state
        if use_tsp_nn_model:
            RLstate = to_RLstate_tsp(state)
        else:
            RLstate = to_RLstate_tsptw(state)

        # Available actions
        avail = env.get_valid_actions(RLstate)
        available = avail.astype(bool)

        # If no available action, the value is 0.0 (no reward can be obtained)
        if np.any(available) == False:
            return 0.0

        # Get GNN input
        graph = env.make_nn_input(RLstate, 'cpu')
        bgraph = dgl.batch([graph])

        # Get RL prediction
        with torch.no_grad():
            res = agent.model(bgraph, graph_pooling=False)
        res = dgl.unbatch(res)

        out = [r.ndata["n_feat"].data.cpu().numpy().flatten() for r in res]
        out = out[0].reshape(-1)

        # Mask unavailable actions
        v = np.max(out[available])

        # Make it minimization
        h = -v
        #print("v", state[unvisited], state[location], v)

        return h

    return RL_heuristic


def create_zero_g_function(location, edges, destination):
    def g_function(parent_g, state):
        cost = edges[state[location], destination]
        g = parent_g + cost

        return g

    return g_function


def create_zero_heuristic_function():
    def zero_heuristic(state):
        return 0

    return zero_heuristic

def create_greedy_heuristic_function(unvisited, location, time_var, 
                                     nodes, edges, a, debug=False):
    distance_matrix = np.array([
        [edges[i, j] if (i, j) in edges else 0 for j in nodes] for i in nodes
    ])

    def greedy_heuristic(state):
        must_visit = list(state[unvisited])
        last_visited = state[location]
        cur_time = state[time_var]

        _must_visit = list(must_visit)
        _cur_time = cur_time

        tour = [last_visited]
        h = 0

        # TSPTW: visit the location with the minimum max(t + c_ij, a_j) and ignore the b_j values
        for i in range(len(_must_visit)):
            arrival_t = np.array(distance_matrix[last_visited]) + _cur_time
            arrival_t = np.array([max(arrival_t[j], a[j]) for j in range(len(arrival_t))])

            # minimum max(t + c_ij, a_j)
            idx = np.argmin(arrival_t[_must_visit])
            next_city = _must_visit[idx]
            tour.append(next_city)
            h += distance_matrix[last_visited][next_city]
            #print("check: i={}, _must_visit={}, arr={}, idx={}, next_city={}, a={}, _cur_time={}".format(
            #    i, _must_visit, arrival_t[_must_visit], idx, next_city, [a[j] for j in _must_visit], _cur_time))

            _cur_time = max(arrival_t[_must_visit][idx], a[next_city])
            last_visited = next_city
            _must_visit.remove(next_city)
        
        # Return to the depot (idx=0)
        h += distance_matrix[last_visited][0]
        
        assert len(_must_visit) == 0
        assert len(set(tour)) == len(tour) # no duplicate cities

        travel_time = sum([distance_matrix[tour[i], tour[i+1]] for i in range(0, len(tour)-1)]
            ) + distance_matrix[tour[-1]][0]
        if debug:
            assert h == travel_time, \
                "h-value ({}) does not match the tour travel time ({})".format(
                    h, travel_time
                )

        # Make sure that the city with minimum distance from the current location is always chosen
        if debug:
            _must_visit = list(must_visit)
            _cur_time = cur_time
            for i in range(len(tour)-1):
                arrival_t = np.array(distance_matrix[tour[i]]) + _cur_time
                arrival_t = np.array([max(arrival_t[j], a[j]) for j in range(len(arrival_t))])
                #print("max({}+{}={}, {})".format(
                #    _cur_time, distance_matrix[tour[i]][tour[i+1]], 
                #    _cur_time + distance_matrix[tour[i]][tour[i+1]],
                #    a[tour[i+1]]))

                assert max(_cur_time + distance_matrix[tour[i]][tour[i+1]], a[tour[i+1]]) == \
                            np.min(arrival_t[_must_visit]), \
                    "The actual arrival time ({}) is not the min arrival time ({})".format(
                        max(_cur_time + distance_matrix[tour[i]][tour[i+1]], a[tour[i+1]]), 
                        np.min(arrival_t[_must_visit])
                    )
                _must_visit.remove(tour[i+1])
                _cur_time = max(_cur_time + distance_matrix[tour[i]][tour[i+1]], a[tour[i+1]])

        return h

    return greedy_heuristic

def create_RL_heuristic_function_debug(
    unvisited, location, time_var, grid_size, max_tw_value, 
    model_folder, input, train_n_city, train_grid_size, train_max_tw_gap, train_max_tw_size, 
    seed, rl_algorithm
):

    def RL_heuristic(state):
        must_visit = state[unvisited]
        last_visited = state[location]
        cur_time = state[time_var]
        #print("heuristic state", must_visit, last_visited, cur_time)
        return 0

    return RL_heuristic


def create_solver(
    model,
    unvisited,
    location,
    customers,
    time_var,
    edges,
    a,
    solver_name,
    scaling_factor=0.01,
    policy_name=None,
    non_zero_base_case=False,
    initial_beam_size=1,
    time_limit=None,
    expansion_limit=None,
    max_beam_size=None,

    # RL heuristic
    heuristic='dual',
    grid_size=None,
    max_tw_value=None,
    heuristic_model_folder=None,
    policy_model_folder=None,
    use_tsp_nn_model=False,
    softmax_temperature=1.0,
    input=None,
    train_n_city=None,
    train_grid_size=None,
    train_max_tw_gap=None,
    train_max_tw_size=None,
    seed=None,
    no_policy_accumulation=True,
    time_window_start=None,
    time_window_end=None
):
    if heuristic == 'dqn':
        assert grid_size is not None
        assert max_tw_value is not None
        assert heuristic_model_folder is not None
        assert input is not None
        assert train_n_city is not None
        assert train_grid_size is not None
        assert train_max_tw_gap is not None
        assert train_max_tw_size is not None
        assert seed is not None
    if policy_name == 'ppo':
        assert grid_size is not None
        assert max_tw_value is not None
        assert policy_model_folder is not None
        assert input is not None
        assert train_n_city is not None
        assert train_grid_size is not None
        assert train_max_tw_gap is not None
        assert train_max_tw_size is not None
        assert seed is not None

    if heuristic != 'dual' and policy_name == 'ppo':
        #max_distance = max(edges[i, j] for i in customers for j in customers if i != j)
        ub_cost = np.sqrt(grid_size ** 2 + grid_size ** 2) * len(customers)
        g_evaluators = {
            "visit {}".format(destination): create_RL_g_function(
                location, edges, destination, ub_cost, scaling_factor
            )
            for destination in customers
        }

        if not non_zero_base_case:
            g_evaluators["return"] = create_RL_g_function(
                location, edges, 0, ub_cost, scaling_factor, use_tsp_nn_model
            )

        h_evaluator = create_RL_heuristic_function(
                unvisited, location, time_var, 
                grid_size, max_tw_value, 
                heuristic_model_folder, input, train_n_city, train_grid_size, 
                train_max_tw_gap, train_max_tw_size, seed, heuristic,
                use_tsp_nn_model
            )

        #policy = create_policy(
        #    unvisited, customers, non_zero_base_case=non_zero_base_case
        #)
        policy = create_RL_policy(unvisited, customers, location, 
                                  time_var, grid_size, max_tw_value, 
                                  policy_model_folder, input, train_n_city, train_grid_size, 
                                  train_max_tw_gap, train_max_tw_size, softmax_temperature,
                                  seed, policy_name, non_zero_base_case=non_zero_base_case,
                                  time_window_start=time_window_start,
                                  time_window_end=time_window_end,
                                  use_tsp_nn_model=use_tsp_nn_model)

        if solver_name == "CABS":
            solver = dp.UserPriorityCABS(
                model,
                h_evaluator=h_evaluator,
                g_evaluators=g_evaluators,
                f_operator=dp.FOperator.Plus,  # g + h
                policy=policy,
                # The user provided heuristic results in negative g and h.
                policy_f_evaluator=policy_f_evaluator_for_negative,
                initial_beam_size=initial_beam_size,
                #max_beam_size=1,
                time_limit=time_limit,
                quiet=False,
                no_policy_accumulation=no_policy_accumulation,
                expansion_limit=expansion_limit
            )
        elif solver_name == "CAASDy":
            solver = dp.UserPriorityCAASDy(
                model,
                h_evaluator=h_evaluator,
                g_evaluators=g_evaluators,
                f_operator=dp.FOperator.Plus,  # g + h
                policy=policy,
                # The user provided heuristic results in negative g and h.
                policy_f_evaluator=policy_f_evaluator_for_negative,
                time_limit=time_limit,
                quiet=False,
                no_policy_accumulation=no_policy_accumulation,
                expansion_limit=expansion_limit
            )
        elif solver_name == "ACPS":
            solver = dp.UserPriorityACPS(
                model,
                h_evaluator=h_evaluator,
                g_evaluators=g_evaluators,
                f_operator=dp.FOperator.Plus,  # g + h
                policy=policy,
                # The user provided heuristic results in negative g and h.
                policy_f_evaluator=policy_f_evaluator_for_negative,
                time_limit=time_limit,
                quiet=False,
                no_policy_accumulation=no_policy_accumulation,
                expansion_limit=expansion_limit
            )
        elif solver_name == "APPS":
            solver = dp.UserPriorityAPPS(
                model,
                h_evaluator=h_evaluator,
                g_evaluators=g_evaluators,
                f_operator=dp.FOperator.Plus,  # g + h
                policy=policy,
                # The user provided heuristic results in negative g and h.
                policy_f_evaluator=policy_f_evaluator_for_negative,
                time_limit=time_limit,
                quiet=False,
                no_policy_accumulation=no_policy_accumulation,
                expansion_limit=expansion_limit
            )

    elif heuristic != 'dual':
        #max_distance = max(edges[i, j] for i in customers for j in customers if i != j)
        ub_cost = np.sqrt(grid_size ** 2 + grid_size ** 2) * len(customers)

        if heuristic == 'dqn':
            g_evaluators = {
                "visit {}".format(destination): create_RL_g_function(
                    location, edges, destination, ub_cost, scaling_factor, use_tsp_nn_model
                )
                for destination in customers
            }

            if not non_zero_base_case:
                g_evaluators["return"] = create_RL_g_function(
                    location, edges, 0, ub_cost, scaling_factor, use_tsp_nn_model
                )

            h_evaluator = create_RL_heuristic_function(
                unvisited, location, time_var, 
                grid_size, max_tw_value, 
                heuristic_model_folder, input, train_n_city, train_grid_size, 
                train_max_tw_gap, train_max_tw_size, seed, heuristic,
                use_tsp_nn_model
            )
        elif heuristic == 'zero':
            g_evaluators = {
                "visit {}".format(destination): create_zero_g_function(
                    location, edges, destination
                    )
                for destination in customers
            }

            if not non_zero_base_case:
                g_evaluators["return"] = create_zero_g_function(location, edges, 0)

            h_evaluator = create_zero_heuristic_function()

        elif heuristic == 'greedy':
            g_evaluators = {
                "visit {}".format(destination): create_zero_g_function(
                    location, edges, destination
                    )
                for destination in customers
            }

            if not non_zero_base_case:
                g_evaluators["return"] = create_zero_g_function(location, edges, 0)

            h_evaluator = create_greedy_heuristic_function(
                unvisited, location, time_var, customers, edges, a, debug=False
                )
        else:
            raise NotImplementedError

        if solver_name == "CAASDy":
            solver = dp.UserPriorityCAASDy(
                model,
                h_evaluator=h_evaluator,
                g_evaluators=g_evaluators,
                f_operator=dp.FOperator.Plus,  # g + h
                time_limit=time_limit,
                quiet=False,
                no_policy_accumulation=no_policy_accumulation,
                expansion_limit=expansion_limit
            )
        elif solver_name == "CABS":
            solver = dp.UserPriorityCABS(
                model,
                h_evaluator=h_evaluator,
                g_evaluators=g_evaluators,
                f_operator=dp.FOperator.Plus,  # g + h
                initial_beam_size=initial_beam_size,
                max_beam_size=max_beam_size,
                time_limit=time_limit,
                quiet=False,
                no_policy_accumulation=no_policy_accumulation,
                expansion_limit=expansion_limit
            )
        elif solver_name == "ACPS":
            solver = dp.UserPriorityACPS(
                model,
                h_evaluator=h_evaluator,
                g_evaluators=g_evaluators,
                f_operator=dp.FOperator.Plus,  # g + h
                time_limit=time_limit,
                quiet=False,
                no_policy_accumulation=no_policy_accumulation,
                expansion_limit=expansion_limit
            )
        elif solver_name == "APPS":
            solver = dp.UserPriorityAPPS(
                model,
                h_evaluator=h_evaluator,
                g_evaluators=g_evaluators,
                f_operator=dp.FOperator.Plus,  # g + h
                time_limit=time_limit,
                quiet=False,
                no_policy_accumulation=no_policy_accumulation,
                expansion_limit=expansion_limit
            )
    elif policy_name == 'ppo':
        policy = create_RL_policy(unvisited, customers, location, 
                                  time_var, grid_size, max_tw_value, 
                                  policy_model_folder, input, train_n_city, train_grid_size, 
                                  train_max_tw_gap, train_max_tw_size, softmax_temperature,
                                  seed, policy_name, non_zero_base_case=non_zero_base_case,
                                  time_window_start=time_window_start,
                                  time_window_end=time_window_end,
                                  use_tsp_nn_model=use_tsp_nn_model)

        if solver_name == "CAASDy":
            solver = dp.UserPriorityCAASDy(
                model,
                policy=policy,
                policy_f_evaluator=policy_f_evaluator,
                time_limit=time_limit,
                quiet=False,
                no_policy_accumulation=no_policy_accumulation,
                expansion_limit=expansion_limit
            )
        elif solver_name == "CABS":
            solver = dp.UserPriorityCABS(
                model,
                policy=policy,
                policy_f_evaluator=policy_f_evaluator,
                initial_beam_size=initial_beam_size,
                time_limit=time_limit,
                quiet=False,
                no_policy_accumulation=no_policy_accumulation,
                expansion_limit=expansion_limit,
                max_beam_size=max_beam_size
            )
        elif solver_name == "ACPS":
            solver = dp.UserPriorityACPS(
                model,
                policy=policy,
                policy_f_evaluator=policy_f_evaluator,
                time_limit=time_limit,
                quiet=False,
                no_policy_accumulation=no_policy_accumulation,
                expansion_limit=expansion_limit
            )
        elif solver_name == "APPS":
            solver = dp.UserPriorityAPPS(
                model,
                policy=policy,
                policy_f_evaluator=policy_f_evaluator,
                time_limit=time_limit,
                quiet=False,
                no_policy_accumulation=no_policy_accumulation,
                expansion_limit=expansion_limit
            )
    elif solver_name == "CAASDy":
        solver = dp.CAASDy(
            model,
            time_limit=time_limit,
            quiet=False,
            expansion_limit=expansion_limit
        )
    elif solver_name == "CABS":
        solver = dp.CABS(
            model,
            initial_beam_size=initial_beam_size,
            time_limit=time_limit,
            quiet=False,
            expansion_limit=expansion_limit,
            max_beam_size=max_beam_size
        )
    elif solver_name == "ACPS":
        solver = dp.ACPS(
            model, 
            time_limit=time_limit, 
            expansion_limit=expansion_limit,
            quiet=False)
    elif solver_name == "APPS":
        solver = dp.APPS(
            model, 
            time_limit=time_limit, 
            expansion_limit=expansion_limit,
            quiet=False)
    else:
        raise NotImplementedError

    return solver

def solve(
    solver,
    name_to_customer,
    history,
    non_zero_base_case=False,
    start_clock_time=None,
    save=1
):
    if start_clock_time is None:
        start_clock_time = time.perf_counter()
    with open(history, "w") as f:
        if save:
            f.write("time, solve_time, expanded, generated, cost, best_bound, is_optimal\n")

        is_terminated = False

        while not is_terminated:
            solution, is_terminated = solver.search_next()

            #debug
            tour = [0]
            for t in solution.transitions:
                tour.append(name_to_customer[t.name])
            if non_zero_base_case:
                tour.append(0)
            #print(tour)

            if save and solution.cost is not None:
                f.write(
                    "{}, {}, {}, {}, {}, {}, {}\n".format(
                        time.perf_counter() - start_clock_time, 
                        solution.time,
                        solution.expanded,
                        solution.generated,
                        solution.cost,
                        solution.best_bound,
                        solution.is_optimal)
                )
                f.flush()

    print("Search time: {}s".format(solution.time))
    print("Expanded: {}".format(solution.expanded))
    print("Generated: {}".format(solution.generated))

    solution_summary = {
        "time": time.perf_counter() - start_clock_time,
        "solve_time": solution.time,
        "expanded": solution.expanded,
        "generated": solution.generated,
        "cost": solution.cost,
        "best_bound": solution.best_bound,
        "is_optimal": solution.is_optimal,
        "is_infeasible": solution.is_infeasible,
    }
    if solution.is_infeasible:
        print("The problem is infeasible")

        return None, None, None
    else:
        tour = [0]

        for t in solution.transitions:
            tour.append(name_to_customer[t.name])

        if non_zero_base_case:
            tour.append(0)

        print(" ".join(map(str, tour[1:-1])))

        print("best bound: {}".format(solution.best_bound))
        print("cost: {}".format(solution.cost))

        if solution.is_optimal:
            print("optimal cost: {}".format(solution.cost))

        return tour, solution.cost, solution_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # DIDP parameters
    parser.add_argument("--policy-name", default="", type=str)
    parser.add_argument("--time-out", default=1800, type=int) # -1 means no time limit
    parser.add_argument("--expansion-limit", default=-1, type=int) # -1 means no node expansion limit
    parser.add_argument("--solver-name", default="CABS", type=str)
    parser.add_argument("--non-zero-base-case", action="store_true")
    parser.add_argument("--threads", default=1, type=int)
    parser.add_argument("--initial-beam-size", default=1, type=int)
    parser.add_argument("--parallel-type", default=0, type=int)
    parser.add_argument('--heuristic', type=str, default='dual') # dual, dqn, ppo, zero
    parser.add_argument('--softmax-temperature', type=float, default=1.0)
    parser.add_argument('--no-policy-accumulation', default=1, type=int)
    parser.add_argument('--use-tsp-nn-model', default=0, type=int)

    # Test instance parameters
    parser.add_argument("--model", default="kuroiwa", type=str)
    parser.add_argument("--instance-type", default="Cappart", type=str)
    parser.add_argument("--n-city", default=20, type=int)
    parser.add_argument("--grid-size", default=100, type=int)
    parser.add_argument("--max-tw-gap", default=10, type=int)
    parser.add_argument("--max-tw-size", default=100, type=int)
    parser.add_argument("--num-instance", default=20, type=int)
    parser.add_argument('--seed', type=int, default=0)

    # RL model parameters
    parser.add_argument('--train-n-city', type=int, default=20)
    parser.add_argument('--train-grid-size', type=int, default=100)
    parser.add_argument('--train-max-tw-gap', type=int, default=10)
    parser.add_argument('--train-max-tw-size', type=int, default=100)
    parser.add_argument('--train-seed', type=int, default=1)

    parser.add_argument('--file', type=str, default="0.txt")

    parser.add_argument('--save', type=int, default=1) # save the solution to a file

    parser.add_argument('--max-beam-size', type=int, default=-1) # for CABS

    args = parser.parse_args()
    if args.heuristic == 'dual' and args.policy_name == 'none': # no nn
        args.use_tsp_nn_model = 0
    if args.time_out == -1:
        args.time_out = None
    if args.expansion_limit == -1:
        args.expansion_limit = None
    if args.max_beam_size == -1:
        args.max_beam_size = None
    print(args)

    if args.heuristic not in ['dual', 'dqn', 'zero', 'greedy']:
        raise NotImplementedError

    if args.instance_type == 'Cappart':
        dataset_path = "%s/n%d/gs%d-tw-%d-%d-ni%d-s%d" % \
                        (args.instance_type, args.n_city, args.grid_size, 
                            args.max_tw_gap, args.max_tw_size,
                            args.num_instance, args.seed)
    elif args.instance_type in ['AFG', 'Dumas', 'GendreauDumasExtended', 'OhlmannThomas']:
        dataset_path = "tsptw/%s" % (args.instance_type)

    load_folder = os.path.join("./instances", "tsptw", dataset_path)
    history_folder = os.path.join(
                        "./history", 
                        "exlim%d" % args.expansion_limit if args.expansion_limit is not None else "no-exlim",
                        args.model, 
                        args.solver_name, 
                        "h-%s-%s" % (args.heuristic, args.policy_name), 
                        "tsptw",
                        "db1", # tsptw models use dual bound
                        "tspnn%d" % args.use_tsp_nn_model,
                        dataset_path
                        )
    print(load_folder)
    assert os.path.exists(load_folder), "The dataset does not exist."
    if not os.path.exists(history_folder):
        os.makedirs(history_folder, exist_ok=True)

    # Create summary.csv with headers
    #if not os.path.exists(os.path.join(history_folder, "summary.csv")):
    #    with open(os.path.join(history_folder, "summary.csv"), "w") as f:
    #        f.write("instance, time, solve_time, expanded, generated, cost, best_bound, is_optimal\n")


    start = time.perf_counter()

    n, nodes, edges, a, b, x_coord, y_coord = read_tsptw.read_xycoords(os.path.join(load_folder, args.file))
    grid_size = math.ceil(max(max(x_coord), max(y_coord)))
    #max_tw_value = max(b)
    max_tw_value = get_max_tw_value(n, args.max_tw_gap, args.max_tw_size)

    if args.model == "kuroiwa":
        model, name_to_customer, unvisited, location, t = create_model_kuroiwa(
            n, nodes, edges, a, b, args.non_zero_base_case
        )
    elif args.model == "cappart":
        args.non_zero_base_case = True
        model, name_to_customer, unvisited, location, t = create_model_cappart(
            n, nodes, edges, a, b
        )
    
    heuristic_model_folder = None
    policy_model_folder = None

    if args.use_tsp_nn_model:
        if args.heuristic == 'dqn':
            heuristic_model_folder = "./cp_rl_solver/selected-models/dqn/tsp/n-city-%d/grid-%d" % \
                            (args.train_n_city, 
                            args.train_grid_size)
        if args.policy_name == 'ppo':
            policy_model_folder = "./cp_rl_solver/selected-models/ppo/tsp/n-city-%d/grid-%d" % \
                            (args.train_n_city,
                            args.train_grid_size)
    else:
        if args.heuristic == 'dqn':
            heuristic_model_folder = "./cp_rl_solver/selected-models/dqn/tsptw/n-city-%d/grid-%d-tw-%d-%d" % \
                            (args.train_n_city, 
                            args.train_grid_size, 
                            args.train_max_tw_gap, 
                            args.train_max_tw_size)
        if args.policy_name == 'ppo':
            policy_model_folder = "./cp_rl_solver/selected-models/ppo/tsptw/n-city-%d/grid-%d-tw-%d-%d" % \
                            (args.train_n_city,
                            args.train_grid_size,
                            args.train_max_tw_gap,
                            args.train_max_tw_size)
        
    no_policy_accumulation = args.no_policy_accumulation == 1

    solver = create_solver(
        model,
        unvisited,
        location,
        nodes,
        t,
        edges,
        a,
        args.solver_name,
        scaling_factor=0.001,
        policy_name=args.policy_name,
        non_zero_base_case=args.non_zero_base_case,
        initial_beam_size=args.initial_beam_size,
        time_limit=args.time_out,
        expansion_limit=args.expansion_limit,
        max_beam_size=args.max_beam_size,
        # RL heuristic
        heuristic=args.heuristic,
        grid_size=grid_size,
        max_tw_value=max_tw_value,
        heuristic_model_folder=heuristic_model_folder,
        policy_model_folder=policy_model_folder,
        use_tsp_nn_model=args.use_tsp_nn_model,
        softmax_temperature=args.softmax_temperature,
        input=os.path.join(load_folder, args.file),
        train_n_city=args.train_n_city,
        train_grid_size=args.train_grid_size,
        train_max_tw_gap=args.train_max_tw_gap,
        train_max_tw_size=args.train_max_tw_size,
        seed=args.train_seed,
        no_policy_accumulation=no_policy_accumulation,
        time_window_start=a,
        time_window_end=b
    )

    tour, cost, summary = solve(
        solver,
        name_to_customer,
        os.path.join(history_folder, "{}.csv".format(args.file.split('.')[0])),
        non_zero_base_case=args.non_zero_base_case,
        start_clock_time=start,
        save=args.save
    )

    if cost is not None and read_tsptw.validate(n, edges, a, b, tour, cost):
        print("The solution is valid.")
    else:
        print("The solution is invalid.")
    
    # Add the solution summary to the summary.csv
    if args.save:
        if summary is not None:
            with open(os.path.join(history_folder, "summary.csv"), "a") as f:
                f.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(
                    args.file.split('.')[0], summary["time"], summary["solve_time"], 
                    summary["expanded"], summary["generated"],
                    summary["cost"], summary["best_bound"], summary["is_optimal"]
                ))
        else:
            with open(os.path.join(history_folder, "summary.csv"), "a") as f:
                f.write("{}, {}, {}, {}, {}, {}, {}\n".format(
                    args.file.split('.')[0], -1, -1, -1, -1, -1, -1
                ))
