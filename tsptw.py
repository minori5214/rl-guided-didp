#!/usr/bin/env python3

import argparse
import time
import os
import math
import copy
import numpy as np

import didppy as dp
from utils.read_tsptw import validate, read_xycoords

import torch
torch.set_num_threads(1)
print("cpu thread used: ", torch.get_num_threads())
os.environ['MKL_NUM_THREADS'] = '1'

def get_max_tw_value(n_city, max_tw_size, max_tw_gap):
    #    :param max_tw_gap: maximum time windows gap allowed between the cities (used for normalization purpose)
    #    :param max_tw_size: time windows of cities will be in the range [0, max_tw_size] (used for normalization purpose)
    return (n_city - 1) * (max_tw_size + max_tw_gap)

def create_DPmodel_kuroiwa(n, nodes, edges, a, b, non_zero_base_case, target=None):
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


def create_DPmodel_cappart(n, nodes, edges, a, b):
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
            g = parent_g + cost * scaling_factor
            assert parent_g <= g

        else:
            # Add the negative cost and scale it
            reward = (ub_cost + (-cost)) * scaling_factor

            # Accumulate negative reward to make it minimization
            g = parent_g + (-reward)

        return g

    return g_function

def create_g_function(location, edges, destination, scaling_factor=1.0):
    def g_function(parent_g, state):
        cost = edges[state[location], destination]
        g = parent_g + cost * scaling_factor
        assert parent_g <= g

        return g

    return g_function

def create_RL_policy(unvisited, customers, location, time_var, grid_size, max_tw_value, 
                        model_folder, input, train_n_city, train_grid_size, 
                        train_max_tw_gap, train_max_tw_size, 
                        softmax_temperature,
                        seed, rl_algorithm, non_zero_base_case=False,
                        use_tsp_nn_model=False):
    """
    A closure function to create an RL policy function.

    Parameters
    ----------
    unvisited : SetVar
        A DIDP state variable for unvisited cities.
    customers : list
        A list of customer indices.
    location : ElementVar
        A DIDP state variable for the current location.
    time_var : IntVar
        A DIDP state variable for the current time.
    grid_size : int
        The grid size of the TSP instance.
    max_tw_value : int
        The maximum time window value.
    model_folder : str
        The folder containing the RL model.
    input : str
        The test instance file name.
    train_n_city : int
        The training instance size.
    train_grid_size : int
        The grid size of the TSP instance used during training.
    train_max_tw_gap : int
        The maximum time windows gap allowed between the cities.
    train_max_tw_size : int
        The time windows of cities will be in the range [0, max_tw_size].
    softmax_temperature : float
        The temperature for the softmax function.
    seed : int
        The random seed used for training.
    rl_algorithm : str
        The name of the RL algorithm. (only 'ppo' is supported)
    non_zero_base_case : bool
        If True, the depot is not included in the unvisited set.
        This setting has to match the setting in the DP model.

    
    """

    import dgl

    if use_tsp_nn_model:
        from tsp_agent import TSPAgent
        from rl_agent.hybrid_cp_rl_solver.problem.tsp.environment.environment import Environment
        from rl_agent.hybrid_cp_rl_solver.problem.tsp.environment.state import State

        # DQN agent
        agent = TSPAgent(model_folder, input, train_n_city, train_grid_size, 
                        seed, rl_algorithm)

        # TSPTW environment
        env = Environment(agent.instance, agent.n_node_feat, agent.n_edge_feat,
                            1, grid_size)
    else:
        from tsptw_agent import TSPTWAgent
        from rl_agent.hybrid_cp_rl_solver.problem.tsptw.environment.environment import Environment
        from rl_agent.hybrid_cp_rl_solver.problem.tsptw.environment.state import State

        # PPO agent
        agent = TSPTWAgent(model_folder, input, train_n_city, train_grid_size, 
                        train_max_tw_gap, train_max_tw_size, seed, rl_algorithm)

        # TSPTW environment
        env = Environment(agent.instance, agent.n_node_feat, agent.n_edge_feat,
                            1, grid_size, max_tw_value)

    def to_RLstate_tsp(DIDPstate):
        must_visit = DIDPstate[unvisited]
        last_visited = DIDPstate[location]

        # State.tour can be left empty
        return State(agent.instance, must_visit, last_visited, tour=[])

    def to_RLstate_tsptw(DIDPstate):
        must_visit = DIDPstate[unvisited]
        last_visited = DIDPstate[location]
        cur_time = DIDPstate[time_var]

        # State.tour can be left empty
        return State(agent.instance, must_visit, last_visited, cur_time, tour=[])

    def policy(state):
        unvisited_set = state[unvisited]
        depot_prob = [np.float64("nan")] 
        if not non_zero_base_case and len(unvisited_set) == 0:
            depot_prob = [1.0] # depot is a forced transition, so must come at the end

        # Convert state to an RL state
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

        action_probs = action_probs + torch.abs(torch.min(action_probs))
        action_probs = action_probs - torch.max(action_probs * available_tensor)

        action_probs = agent.actor_critic_network.masked_softmax(
                            action_probs, available_tensor, dim=0,
                            temperature=softmax_temperature)

        # If the action is available, and the probability is 0.0, we set it to the minimum possible float
        action_probs = action_probs + (1e-32 * (available_tensor == 1.0))

        action_probs = action_probs.detach()

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

        # Policy must return log probabilities 
        # (to match the f_evaluator implementation in didp-rs)
        log_probabilities = np.log(probabilities)

        return log_probabilities

    return policy

def policy_f_evaluator(log_pi, g, h):
    if h is None:
        h = 0.0 # if there is no dual bound, only g is used

    # PHS priority function in logarithm
    f = np.log(g + h) - log_pi
    return f

def policy_f_evaluator_for_negative(log_pi, g, h):
    # If g- and h-values are negative, we take the negative to make it positive.
    # Then, it becomes maximization, so we multiply pi (or add log pi to the logarithm)
    # and take the negative to put it back to minimization
    f = -(np.log(-(g + h)) + log_pi)
    return f

def create_RL_heuristic_function(
    unvisited, location, time_var, grid_size, max_tw_value, 
    model_folder, input, train_n_city, train_grid_size, train_max_tw_gap, train_max_tw_size, 
    seed, rl_algorithm, use_tsp_nn_model
):
    """
    A closure function to create an RL heuristic function.

    Parameters
    ----------
    unvisited : SetVar
        A DIDP state variable for unvisited cities.
    location : ElementVar
        A DIDP state variable for the current location.
    time_var : IntVar
        A DIDP state variable for the current time.
    grid_size : int
        The grid size of the TSP instance.
    max_tw_value : int
        The maximum time window value.
    model_folder : str
        The folder containing the RL model.
    input : str
        The test instance file name.
    train_n_city : int
        The training instance size.
    train_grid_size : int
        The grid size of the TSP instance used during training.
    train_max_tw_gap : int
        The maximum time windows gap allowed between the cities.
    train_max_tw_size : int
        The time windows of cities will be in the range [0, max_tw_size].
    seed : int
        The random seed used for training.
    rl_algorithm : str
        The name of the RL algorithm. (only 'dqn' is supported)
    use_tsp_nn_model : bool
        If True, the TSP RL model is used.
    
    """
    import dgl

    if use_tsp_nn_model:
        from tsp_agent import TSPAgent
        from rl_agent.hybrid_cp_rl_solver.problem.tsp.environment.environment import Environment
        from rl_agent.hybrid_cp_rl_solver.problem.tsp.environment.state import State

        # DQN agent
        agent = TSPAgent(model_folder, input, train_n_city, train_grid_size, 
                        seed, rl_algorithm)

        # TSPTW environment
        env = Environment(agent.instance, agent.n_node_feat, agent.n_edge_feat,
                            1, grid_size)

    else:
        from tsptw_agent import TSPTWAgent
        from rl_agent.hybrid_cp_rl_solver.problem.tsptw.environment.environment import Environment
        from rl_agent.hybrid_cp_rl_solver.problem.tsptw.environment.state import State

        # DQN agent
        agent = TSPTWAgent(model_folder, input, train_n_city, train_grid_size, 
                        train_max_tw_gap, train_max_tw_size, seed, rl_algorithm)

        # TSPTW environment
        env = Environment(agent.instance, agent.n_node_feat, agent.n_edge_feat,
                            1, grid_size, max_tw_value)

    def to_RLstate_tsp(DIDPstate):
        must_visit = DIDPstate[unvisited]
        last_visited = DIDPstate[location]

        # State.tour can be left empty
        return State(agent.instance, must_visit, last_visited, tour=[])

    def to_RLstate_tsptw(DIDPstate):
        must_visit = DIDPstate[unvisited]
        last_visited = DIDPstate[location]
        cur_time = DIDPstate[time_var]

        # State.tour can be left empty
        return State(agent.instance, must_visit, last_visited, cur_time, tour=[])

    def RL_heuristic(state):
        # Convert state to an RL state
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

        # Get a GNN input
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

        return h

    return RL_heuristic

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

def create_zero_heuristic_function():
    def zero_heuristic(state):
        return 0

    return zero_heuristic

def create_solver(
    # DIDP parameters
    model, # DIDP model
    unvisited, # state variable
    location, # state variable
    time_var, # state variable
    customers, # (list) customer indices
    edges, # c_ij (dict)
    a, # earliest arrival time,
    solver_name, # CABS, ACPS, APPS
    non_zero_base_case=False,
    initial_beam_size=1,
    time_limit=None,
    heuristic='dual', # dual, dqn, zero, greedy
    policy_name=None, # None, ppo
    # RL parameters
    use_tsp_nn_model=False, # True if the TSP RL model is used
    scaling_factor=0.001,
    grid_size=None,
    heuristic_model_folder=None,
    policy_model_folder=None,
    softmax_temperature=1.0,
    input=None, # test instance file
    train_n_city=None,
    train_grid_size=None,
    train_max_tw_gap=None,
    train_max_tw_size=None,
    max_tw_value=None,
    seed=None,
    no_policy_accumulation=False
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

    if heuristic == 'dual' and policy_name is None:
        solvers = {
            "CABS": dp.CABS,
            "CAASDy": dp.CAASDy,
            "ACPS": dp.ACPS,
            "APPS": dp.APPS
        }
        return solvers.get(solver_name, None)(
                    model, 
                    time_limit=time_limit)

    def create_g_evaluators(scaling_factor):
        """
        Create g evaluators for each destination city.
        User-defined g-evaluators are needed to use the user-defined h-value function.
        g-evaluator for returning to the depot is also created 
        if non_zero_base_case is False.

        Parameters
        ----------
        scaling_factor : float
            Scaling factor for the g-value.
            This value has to match the scaling factor used during the RL training.
            For zero and greedy heuristics, no scaling is needed (= 1.0).
        
        """
        g_evaluators = {}
        if heuristic in ['zero', 'greedy']:
            scaling_factor = 1.0

            for dest in customers:
                g_evaluators[f"visit {dest}"] = \
                    create_g_function(location, edges, dest, scaling_factor=scaling_factor)

            if not non_zero_base_case:
                g_evaluators["return"] = create_g_function(location, edges, 0, scaling_factor)

        elif heuristic == 'dqn':
            ub_cost = np.sqrt(grid_size ** 2 + grid_size ** 2) * len(customers)
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
        else:
            raise NotImplementedError

        return g_evaluators

    def create_h_evaluator():
        """
        Create h evaluator based on the heuristic function.
        
        """

        if heuristic == 'dqn':
            return create_RL_heuristic_function(
                unvisited, 
                location, 
                time_var, 
                grid_size, 
                max_tw_value, 
                heuristic_model_folder, 
                input, 
                train_n_city, 
                train_grid_size, 
                train_max_tw_gap, 
                train_max_tw_size, 
                seed, 
                heuristic,
                use_tsp_nn_model
            )

        elif heuristic == 'zero':
            return create_zero_heuristic_function()
        elif heuristic == 'greedy':
            return create_greedy_heuristic_function(
                        unvisited, 
                        location, 
                        time_var,
                        customers, 
                        edges, 
                        a,
                        debug=False
                        )
        else:
            raise NotImplementedError

    def create_solver_instance(solver_name, **kwargs):
        solvers = {
            "CABS": dp.UserPriorityCABS,
            "CAASDy": dp.UserPriorityCAASDy,
            "ACPS": dp.UserPriorityACPS,
            "APPS": dp.UserPriorityAPPS
        }
        return solvers.get(solver_name, None)(model, **kwargs)

    if heuristic != 'dual':
        g_evaluators = create_g_evaluators(scaling_factor=scaling_factor)
        h_evaluator = create_h_evaluator()

    if policy_name == 'ppo':
        policy = create_RL_policy(
                        unvisited, 
                        customers, 
                        location, 
                        time_var, 
                        grid_size, 
                        max_tw_value, 
                        policy_model_folder, 
                        input, 
                        train_n_city, 
                        train_grid_size, 
                        train_max_tw_gap, 
                        train_max_tw_size, 
                        softmax_temperature,
                        seed, 
                        policy_name, 
                        non_zero_base_case=non_zero_base_case,
                        use_tsp_nn_model=use_tsp_nn_model
                        )

    solver_params = {
        "h_evaluator": h_evaluator if heuristic != 'dual' else None,
        "g_evaluators": g_evaluators if heuristic != 'dual' else None,
        "f_operator": dp.FOperator.Plus if heuristic != 'dual' else None,
        "policy": policy if policy_name == 'ppo' else None,
        "policy_f_evaluator": policy_f_evaluator if policy_name == 'ppo' else None,
        "initial_beam_size": initial_beam_size if solver_name == "CABS" else None,
        "time_limit": time_limit,
        "quiet": False,
        "no_policy_accumulation": no_policy_accumulation if policy_name == 'ppo' else None
    }

    return create_solver_instance(
                solver_name, 
                **{k: v for k, v in solver_params.items() if v is not None}
                )

def solve(
    solver,
    name_to_customer,
    non_zero_base_case=False,
    start_clock_time=None,
):
    if start_clock_time is None:
        start_clock_time = time.perf_counter()

    is_terminated = False

    while not is_terminated:
        solution, is_terminated = solver.search_next()

        tour = [0]
        for t in solution.transitions:
            tour.append(name_to_customer[t.name])
        if non_zero_base_case:
            tour.append(0)

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
    parser.add_argument("--model", default="kuroiwa", type=str) # DIDP model (kuroiwa or cappart)
    parser.add_argument("--solver-name", default="CABS", type=str) # CABS, ACPS, APPS, CAASDy
    parser.add_argument('--heuristic', type=str, default='dual') # dual, dqn, zero, greedy
    parser.add_argument("--policy-name", default="none", type=str) # None, ppo
    parser.add_argument("--time-out", default=1800, type=int) # seconds, -1 means no time limit
    parser.add_argument("--non-zero-base-case", action="store_true")
    parser.add_argument("--threads", default=1, type=int)
    parser.add_argument("--initial-beam-size", default=1, type=int) # initial beam size for CABS, ACPS, APPS

    # Test instance parameters
    parser.add_argument("--n-city", default=20, type=int) # problem size
    parser.add_argument("--grid-size", default=100, type=int) # instance distribution parameter
    parser.add_argument("--max-tw-gap", default=100, type=int) # instance distribution parameter
    parser.add_argument("--max-tw-size", default=1000, type=int) # instance distribution parameter

    parser.add_argument("--num-instance", default=100, type=int)
    parser.add_argument('--seed', type=int, default=0)

    # RL model parameters
    parser.add_argument('--use-tsp-nn-model', default=0, type=int) # 1 if the TSP RL model is used
    parser.add_argument('--scaling-factor', type=float, default=0.001) # scaling factor of the reward function
    parser.add_argument('--softmax-temperature', type=float, default=1.0) # temperature for ppo
    parser.add_argument('--no-policy-accumulation', default=0, type=int) # 0 if accumulated policy from the root node is used for guidance
    parser.add_argument('--train-n-city', type=int, default=20) # training instance size
    parser.add_argument('--train-grid-size', type=int, default=100) # instance distribution parameter
    parser.add_argument('--train-max-tw-gap', type=int, default=100)
    parser.add_argument('--train-max-tw-size', type=int, default=1000)
    parser.add_argument('--train-seed', type=int, default=1) # random seed used for training

    parser.add_argument('--file', type=str, default="0.txt") # test instance file

    args = parser.parse_args()
    if args.time_out == -1:
        args.time_out = None
    if args.policy_name == "none":
        args.policy_name = None
    no_policy_accumulation = args.no_policy_accumulation == 1

    print(args)

    if args.heuristic not in ['dual', 'dqn', 'zero', 'greedy']:
        raise NotImplementedError

    dataset_path = "Cappart/n%d/gs%d-tw-%d-%d-ni%d-s%d" % \
                    (args.n_city, args.grid_size, 
                        args.max_tw_gap, args.max_tw_size,
                        args.num_instance, args.seed)
    load_folder = os.path.join("./instances", "tsptw", dataset_path)

    print("load_folder", load_folder)
    assert os.path.exists(load_folder), "The dataset does not exist."

    start = time.perf_counter()

    # Read TSPTW instance
    n, nodes, edges, a, b, x_coord, y_coord = \
        read_xycoords(os.path.join(load_folder, args.file))

    grid_size = math.ceil(max(max(x_coord), max(y_coord)))
    max_tw_value = get_max_tw_value(n, args.max_tw_gap, args.max_tw_size)

    # Define the DP model
    if args.model == "kuroiwa":
        model, name_to_customer, unvisited, location, t = create_DPmodel_kuroiwa(
            n, nodes, edges, a, b, args.non_zero_base_case
        )
    elif args.model == "cappart":
        args.non_zero_base_case = True
        model, name_to_customer, unvisited, location, t = create_DPmodel_cappart(
            n, nodes, edges, a, b
        )


    if args.use_tsp_nn_model:
        # DQN model folder
        heuristic_model_folder = \
            "./rl_agent/hybrid_cp_rl_solver/selected-models/dqn/tsp/n-city-%d/grid-%d" % \
                        (args.train_n_city, 
                        args.train_grid_size)
        # PPO model folder
        policy_model_folder = \
            "./rl_agent/hybrid_cp_rl_solver/selected-models/ppo/tsp/n-city-%d/grid-%d" % \
                        (args.train_n_city,
                        args.train_grid_size)
    else:
        # DQN model folder
        heuristic_model_folder = \
            "./rl_agent/hybrid_cp_rl_solver/selected-models/dqn/tsptw/n-city-%d/grid-%d-tw-%d-%d" % \
                        (args.train_n_city, 
                        args.train_grid_size,
                        args.train_max_tw_gap,
                        args.train_max_tw_size)
        # PPO model folder
        policy_model_folder = \
            "./rl_agent/hybrid_cp_rl_solver/selected-models/ppo/tsptw/n-city-%d/grid-%d-tw-%d-%d" % \
                        (args.train_n_city,
                        args.train_grid_size,
                        args.train_max_tw_gap,
                        args.train_max_tw_size)

    solver = create_solver(
        # DIDP parameters
        model,
        unvisited,
        location,
        t,
        nodes,
        edges,
        a,
        args.solver_name,
        non_zero_base_case=args.non_zero_base_case,
        initial_beam_size=args.initial_beam_size,
        time_limit=args.time_out,
        heuristic=args.heuristic,
        policy_name=args.policy_name,
        # RL parameters
        use_tsp_nn_model=args.use_tsp_nn_model,
        scaling_factor=args.scaling_factor,
        grid_size=grid_size,
        heuristic_model_folder=heuristic_model_folder,
        policy_model_folder=policy_model_folder,
        softmax_temperature=args.softmax_temperature,
        input=os.path.join(load_folder, args.file),
        train_n_city=args.train_n_city,
        train_grid_size=args.train_grid_size,
        train_max_tw_gap=args.train_max_tw_gap,
        train_max_tw_size=args.train_max_tw_size,
        max_tw_value=max_tw_value,
        seed=args.train_seed,
        no_policy_accumulation=no_policy_accumulation,
        )

    tour, cost, summary = solve(
        solver,
        name_to_customer,
        non_zero_base_case=args.non_zero_base_case,
        start_clock_time=start,
    )

    if cost is not None and validate(n, edges, a, b, tour, cost):
        print("The solution is valid.")
    else:
        print("The solution is invalid.")