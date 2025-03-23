#!/usr/bin/env python3

import argparse
import time
import os
import numpy as np

import didppy as dp
from utils.read_portfolio import read, validate

import torch
torch.set_num_threads(1)
print("cpu thread used: ", torch.get_num_threads())
os.environ['MKL_NUM_THREADS'] = '1'


def create_DPmodel_narita(n, capacity, w, m, d, s, k, lmd, discrete_coeff=0, 
                        dual_bound=2, target=None):
    m = m + [0]
    d = d + [0]
    s = s + [0]
    k = k + [0]
    if target is None:
        target_total_weight = 0
        target_stage = 0
        target_invested = [0 for _ in range(n)]
        target_packed = []
    else:
        assert type(target) == list
        assert len(target) == 3
        assert type(target[0]) == int or type(target[0]) == float
        assert type(target[1]) == int
        assert type(target[2]) == list
        target_total_weight = target[0]
        target_stage = target[1]
        target_invested = target[2]

    if discrete_coeff == 1:
        model = dp.Model(maximize=True)
    else:
        model = dp.Model(maximize=True, float_cost=True)

    investment = model.add_object_type(number=n)

    # c (total weight)
    if discrete_coeff == 1:
        total_weight = model.add_int_var(target=target_total_weight)
    else:
        total_weight = model.add_float_var(target=target_total_weight)

    # target cost is the total cost so far at stage i
    target_cost = lmd[0]*sum([m[j]*target_invested[j] for j in range(target_stage)])-\
        lmd[1]*(sum([d[j]*target_invested[j] for j in range(target_stage)])**(1/2))+\
        lmd[2]*(sum([s[j]*target_invested[j] for j in range(target_stage)])**(1/3))-\
        lmd[3]*(sum([k[j]*target_invested[j] for j in range(target_stage)])**(1/4))
    if discrete_coeff == 1:
        total_cost = model.add_int_var(target=int(target_cost))
    else:
        total_cost = model.add_float_var(target=target_cost)

    # i (current item)
    i = model.add_element_var(object_type=investment, target=target_stage)

    packed = model.add_set_var(object_type=investment, target=target_packed)

    weights = model.add_int_table(w)
    means = model.add_int_table(m)
    deviations = model.add_int_table(d)
    skewnesses = model.add_int_table(s)
    kurtosis = model.add_int_table(k)
    lambdas = model.add_int_table(lmd)

    if discrete_coeff == 1:
        _cost = dp.IntExpr.state_cost()
    else:
        _cost = dp.FloatExpr.state_cost()

    ignore = dp.Transition(
        name="ignore",
        cost= _cost,
        effects=[(i, i + 1)],
        preconditions=[i < n],
    )
    model.add_transition(ignore)

    invest = dp.Transition(
        name="invest",
        cost=lambdas[0]*(means[packed] + means[i])-\
                lambdas[1]*((deviations[packed] + deviations[i])**(1/2))+\
                lambdas[2]*((skewnesses[packed] + skewnesses[i])**(1/3))-\
                lambdas[3]*((kurtosis[packed] + kurtosis[i])**(1/4)),
        effects=[
            (total_weight, total_weight + weights[i]),
            (i, i + 1),
            (packed, packed.add(i))
            ],
        preconditions=[
            i < n,
            total_weight + weights[i] <= capacity
            ],
    )
    model.add_transition(invest)
    print("dual bound", dual_bound)

    if dual_bound >= 1:
        # Assume we take all the remaining items
        remaining_m = lambdas[0]*sum([(i <= j).if_then_else(means[j], 0) for j in range(n)])
        remaining_s = lambdas[2]*(sum([(i <= j).if_then_else(skewnesses[j], 0) for j in range(n)])**(1/3))
        model.add_dual_bound(remaining_m + remaining_s)

    if dual_bound >= 2:
        # Max efficiency of the remaining items considering only the positive terms
        # For each stage j, Max efficiency (j) * remaining capacity
        max_efficiency = [max((lmd[0]*m[j] + lmd[2]*s[j]**(1/3)) \
                              / (w[j] + 1e-6) for j in range(q, n)) 
                            for q in range(n)] + [0] # [0] is for the base case
        max_eff = model.add_float_table(max_efficiency)
        model.add_dual_bound(max_eff[i] * (capacity - total_weight))

    model.add_base_case([i == n])

    return model, total_weight, i, packed, means, deviations, skewnesses, kurtosis, lambdas

def create_RL_g_function(scaling_factor, total_weight, stage, is_pack, packed, 
                         means, deviations, skewnesses, kurtosis, lambdas,
                         model_name='narita'):
    def g_function(parent_g, state):
        if model_name == 'narita':
            g = lambdas[0]*(sum([means[j] for j in state[packed]])+is_pack*means[state[stage]])-\
                    lambdas[1]*((sum([deviations[j] for j in state[packed]])+is_pack*deviations[state[stage]])**(1/2))+\
                    lambdas[2]*((sum([skewnesses[j] for j in state[packed]])+is_pack*skewnesses[state[stage]])**(1/3))-\
                    lambdas[3]*((sum([kurtosis[j] for j in state[packed]])+is_pack*kurtosis[state[stage]])**(1/4))
            g *= scaling_factor

            # Need to negate the g-value to make it minimization 
            # (as user-heuristic search always performs minimization)
            g = -g
            #print("138 g", g)

        else:
            raise NotImplementedError

        return g

    return g_function

def create_g_function(total_weight, stage, is_pack, packed, 
                         means, deviations, skewnesses, kurtosis, lambdas,
                         model_name='narita'):

    def g_function(parent_g, state):
        if model_name == 'narita':
            g = lambdas[0]*(sum([means[j] for j in state[packed]])+is_pack*means[state[stage]])-\
                    lambdas[1]*((sum([deviations[j] for j in state[packed]])+is_pack*deviations[state[stage]])**(1/2))+\
                    lambdas[2]*((sum([skewnesses[j] for j in state[packed]])+is_pack*skewnesses[state[stage]])**(1/3))-\
                    lambdas[3]*((sum([kurtosis[j] for j in state[packed]])+is_pack*kurtosis[state[stage]])**(1/4))
        else:
            raise NotImplementedError

        # User-heuristic searcb always perform minimization;
        # thus, we need to negate the value if the original problem is for maximization
        return -g

    return g_function

def create_RL_policy(model, 
                    total_weight, stage, packed, 
                    n_item, capacity, weights, 
                    model_folder, input, train_n_item, train_capacity_ratio,
                    train_lambdas_0, train_lambdas_1, 
                    train_lambdas_2, train_lambdas_3,
                    discrete_coeff, softmax_temperature, seed, 
                    rl_algorithm='ppo'):
    """
    A closure function to create an RL policy function.

    Parameters
    ----------
    model : str
        The name of the DIDP model. (only 'narita' is supported)
    total_weight : SetVar
        A DIDP state variable for the total weight of the portfolio.
    stage : SetVar
        A DIDP state variable for the current stage.
    packed : SetVar
        A DIDP state variable for the packed items.
    n_item : int
        Problem size.
    capacity : int
        The capacity of the portfolio.
    weights : list
        A list of weights of the items.
    model_folder : str
        The folder containing the RL model.
    input : str
        The test instance file name.
    train_n_item : int
        The training instance size.
    train_capacity_ratio : float
        The training instance capacity ratio.
    train_lambdas_X : int
        The Xth moment factor used during training.
    discrete_coeff : int
        1 if the cost is discrete or not, 0 otherwise.
    softmax_temperature : float
        The temperature for the softmax function.
    seed : int
        The random seed used for training.
    rl_algorithm : str
        The name of the RL algorithm. (only 'ppo' is supported)
    
    """
    from portfolio_agent import PortfolioAgent
    from rl_agent.hybrid_cp_rl_solver.problem.portfolio.environment.environment import Environment
    from rl_agent.hybrid_cp_rl_solver.problem.portfolio.environment.state import State

    # PPO agent
    agent = PortfolioAgent(model_folder, input, train_n_item, train_capacity_ratio,
                 train_lambdas_0, train_lambdas_1, 
                 train_lambdas_2, train_lambdas_3,
                 discrete_coeff, seed, rl_algorithm)

    # Portfolio environment
    env = Environment(agent.instance, agent.n_feat, 1)

    def to_RLstate(DIDPstate):
        weight = DIDPstate[total_weight]
        cur_stage = DIDPstate[stage]

        # Not possible to insert the item if its insertion exceed the portfolio capacity
        if cur_stage == n_item:
            available_action = set()
        elif weight + weights[cur_stage] > capacity:
            available_action = set([0])
        else:
            available_action = set([0, 1])
        
        if model == 'narita':
            items_taken = [1 for j in range(agent.n_item) if j in DIDPstate[packed]]
        else:
            raise NotImplementedError

        # State.tour can be left empty
        return State(agent.instance, weight, cur_stage, available_action, items_taken)

    def policy(state):
        # Convert state to an RL state
        RLstate = to_RLstate(state)

        state_feats = env.make_nn_input(RLstate, "cpu")
        avail = env.get_valid_actions(RLstate)

        available_tensor = torch.FloatTensor(avail)

        with torch.no_grad():
            batched_set = state_feats.unsqueeze(0)
            out = agent.model(batched_set)
            action_probs = out.squeeze(0)

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

        # If the action is not available, we set the probability to np.nan

        probabilities = np.array(
            [
                action_probs[j]
                for j in range(len(avail))
            ],
            dtype=np.float64,
        )

        # Policy must return log probabilities 
        # (to match the f_evaluator implementation in didp-rs)
        log_probabilities = np.log(probabilities)

        return log_probabilities

    return policy

def policy_f_evaluator_maximization(log_pi, g, h):
    if h is None:
        h =0.0 # if there is no dual bound, only g is used

    # PHS priority function in logarithm
    if (g + h) == 0:
        return -log_pi

    # User-heuristic searcb always perform minimization;
    # thus, we need to negate the value if the original problem is for maximization
    # (e.g., the quality if better if np.log(g + h) + log_pi is higher -> negate the value)
    f = -(np.log(g + h) + log_pi)
    return f

def policy_f_evaluator(log_pi, g, h):
    """
    The f-evaluator function for user-heuristics.
    g and h-values are supporsed to be for minimization, and thus
    they are negated to make them maximization.
    
    """

    if h is None:
        h =0.0 # if there is no dual bound, only g is used

    # PHS priority function in logarithm
    if ((-g) + (-h)) == 0:
        return -log_pi

    #f = np.log(g + h) - log_pi # Wrong
    f = -(np.log((-g) + (-h)) + log_pi)
    return f

def create_RL_heuristic_function(
    model, total_weight, stage, packed, 
    n_item, capacity, weights, 
    model_folder, input, train_n_item, train_capacity_ratio,
    train_lambdas_0, train_lambdas_1, 
    train_lambdas_2, train_lambdas_3,
    discrete_coeff, scaling_factor, seed, rl_algorithm
):
    """
    A closure function to create an RL heuristic function.

    Parameters
    ----------
    model : str
        The name of the DIDP model.
    total_weight : SetVar
        A DIDP state variable for the total weight of the portfolio.
    stage : SetVar
        A DIDP state variable for the current stage.
    packed : SetVar
        A DIDP state variable for the packed items.
    n_item : int
        Problem size.
    capacity : int
        The capacity of the portfolio.
    weights : list
        A list of weights of the items.
    model_folder : str
        The folder containing the RL model.
    input : str
        The test instance file name.
    train_n_item : int
        The training instance size.
    train_capacity_ratio : float
        The training instance capacity ratio.
    train_lambdas_X : int
        The Xth moment factor used during training.
    discrete_coeff : int
        1 if the cost is discrete or not, 0 otherwise.
    scaling_factor : float
        Scaling factor for the heuristic value.
    seed : int
        The random seed used for training.
    rl_algorithm : str
        The name of the RL algorithm. (only 'dqn' is supported)
    
    """

    from portfolio_agent import PortfolioAgent
    from rl_agent.hybrid_cp_rl_solver.problem.portfolio.environment.environment import Environment
    from rl_agent.hybrid_cp_rl_solver.problem.portfolio.environment.state import State

    # DQN agent
    agent = PortfolioAgent(model_folder, input, train_n_item, train_capacity_ratio,
                 train_lambdas_0, train_lambdas_1, 
                 train_lambdas_2, train_lambdas_3,
                 discrete_coeff, seed, rl_algorithm)

    # Portfolio environment
    env = Environment(agent.instance, agent.n_feat, 1)

    def to_RLstate(DIDPstate):
        weight = DIDPstate[total_weight]
        cur_stage = DIDPstate[stage]

        # Not possible to insert the item if its insertion exceed the portfolio capacity
        if cur_stage == n_item:
            available_action = set()
        elif weight + weights[cur_stage] > capacity:
            available_action = set([0])
        else:
            available_action = set([0, 1])
        
        if model == 'narita':
            items_taken = [1 for j in range(agent.n_item) if j in DIDPstate[packed]]
        else:
            raise NotImplementedError

        # State.tour can be left empty
        return State(agent.instance, weight, cur_stage, available_action, items_taken)

    def RL_heuristic(state):
        # Convert state to an RL state
        RLstate = to_RLstate(state)

        # Available actions
        avail = env.get_valid_actions(RLstate)
        available = avail.astype(bool)

        # If no available action, it is the base case - calculate the true value
        if np.any(available) == False:
            if model == 'narita':
                cost = 0.0
            else:
                raise NotImplementedError
            v = cost * scaling_factor
            h = -v
            return h

        nn_input = env.make_nn_input(RLstate, 'cpu')
        nn_input = nn_input.unsqueeze(0)

        # Get RL prediction
        with torch.no_grad():
            res = agent.model(nn_input)

        out = res.cpu().numpy().squeeze(0)

        if rl_algorithm == 'dqn':
            # Mask unavailable actions
            v = np.max(out[available])
        elif rl_algorithm == 'ppoval':
            v = out[0]

        # h-evaluator returns the h-value for minimization
        h = -v
        #print("h", h)

        return h

    return RL_heuristic


def create_zero_heuristic_function():
    def zero_heuristic(state):
        return 0

    return zero_heuristic

def create_greedy_heuristic_function(n, weights, capacity, 
                                     m, d, s, k, lmd,
                                     total_weight, stage, 
                                     heuristic_type=1, debug=True):
    # heuristic_type (int): 1: use only means and skewness (same as dual bound), 
    #                       2: use deviations too, 
    #                       3: use kurtosis too, 
    #                       4: use all

    assert heuristic_type in [1, 2, 3, 4], "Invalid heuristic type ({})".format(heuristic_type)

    if heuristic_type == 1:
        d = [0 for _ in range(n)]
        k = [0 for _ in range(n)]
    elif heuristic_type == 2:
        k = [0 for _ in range(n)]
    elif heuristic_type == 3:
        d = [0 for _ in range(n)]

    efficiency = np.array([(lmd[0]*m[j] - lmd[1]*d[j]**(1/2) + lmd[2]*s[j]**(1/3) + lmd[3]*k[j]**(1/4)) \
                    / (weights[j] + 1e-6) for j in range(n)] + [0]) # [0] is for the base case

    def greedy_heuristic(state):
        # Portfolio: selecting the item with the highest 
        # "mean - deviation + skewness - kurtosis" divided by its weight

        cur_weight = state[total_weight]
        cur_stage = state[stage]

        _cur_weight = cur_weight
        _cur_stage = cur_stage

        solution = []

        # Consider each item in the remaining items (_cur_stage to n), 
        # and choose the best ratio item until the capacity is full
        remaining_items = [i for i in range(_cur_stage, n)]

        # Order the items by the efficiency
        _idxs = np.argsort(efficiency[remaining_items])[::-1]
        items_bestratio = [remaining_items[j] for j in _idxs]

        # Select the item with the highest efficiency, if the weight is not exceeded
        for j in items_bestratio:
            if _cur_weight + weights[j] <= capacity:
                solution.append(j)
                _cur_weight += weights[j]
        
        h = lmd[0] * sum([m[j] for j in solution]) -\
            lmd[1] * (sum([d[j] for j in solution])**(1/2)) +\
            lmd[2] * (sum([s[j] for j in solution])**(1/3)) -\
            lmd[3] * (sum([k[j] for j in solution])**(1/4))

        assert _cur_weight <= capacity, "The solution exceeds the capacity"
        assert len(solution) == len(set(solution)), "The solution contains duplicate items. solution={}".format(solution)

        return -h # negate the h-value to keep it minimization

    return greedy_heuristic

def create_solver(
    # DIDP parameters
    model, # DIDP model
    total_weight, # state variable (x)
    stage, # state variable (i)
    packed, # state variable (Y)
    means, # m
    deviations, # sigma
    skewnesses, # gamma
    kurtosis, # kappa
    lambdas, # lambda
    n_item,
    capacity,
    weights, # weight of items (list)
    solver_name, # CABS, ACPS, APPS
    discrete_coeff=0,
    greedy_heuristic_type=4,
    initial_beam_size=1,
    time_limit=None,
    heuristic='dual', # dual, dqn, zero, greedy
    policy_name=None, # None, ppo
    # RL parameters
    dp_model_name='narita',
    scaling_factor=0.001,
    heuristic_model_folder=None,
    policy_model_folder=None,
    softmax_temperature=1.0,
    input=None, # test instance file
    train_n_item=None,
    train_capacity_ratio=None,
    train_lambdas_0=None,
    train_lambdas_1=None,
    train_lambdas_2=None,
    train_lambdas_3=None,
    seed=None,
    no_policy_accumulation=False
):
    if heuristic == 'dqn':
        assert heuristic_model_folder is not None
        assert input is not None
        assert train_n_item is not None
        assert train_capacity_ratio is not None
        assert train_lambdas_0 is not None
        assert train_lambdas_1 is not None
        assert train_lambdas_2 is not None
        assert train_lambdas_3 is not None
        assert seed is not None
    if policy_name == 'ppo':
        assert policy_model_folder is not None
        assert input is not None
        assert train_n_item is not None
        assert train_capacity_ratio is not None
        assert train_lambdas_0 is not None
        assert train_lambdas_1 is not None
        assert train_lambdas_2 is not None
        assert train_lambdas_3 is not None
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

        Parameters
        ----------
        scaling_factor : float
            Scaling factor for the g-value.
            This value has to match the scaling factor used during the RL training.
            For zero and greedy heuristics, no scaling is needed (= 1.0).
        
        """
        if heuristic in ['zero', 'greedy']:
            scaling_factor = 1.0

            g_evaluators = {}
            g_evaluators["invest"] = create_g_function(
                    total_weight, stage, 1, packed, 
                    means, deviations, skewnesses, kurtosis, lambdas,
                    model_name=dp_model_name
                )
            g_evaluators["ignore"] = create_g_function(
                    total_weight, stage, 0, packed, 
                    means, deviations, skewnesses, kurtosis, lambdas,
                    model_name=dp_model_name
                )

        elif heuristic == 'dqn':
            g_evaluators = {}
            g_evaluators["invest"] = create_RL_g_function(
                    scaling_factor, total_weight, stage, 1, packed, 
                    means, deviations, skewnesses, kurtosis, lambdas,
                    model_name=dp_model_name
                )
            g_evaluators["ignore"] = create_RL_g_function(
                    scaling_factor, total_weight, stage, 0, packed, 
                    means, deviations, skewnesses, kurtosis, lambdas,
                    model_name=dp_model_name
                )
        
        return g_evaluators

    def create_h_evaluator():
        """
        Create h evaluator based on the heuristic function.
        
        """

        if heuristic == 'dqn':
            return create_RL_heuristic_function(
                dp_model_name, 
                total_weight, stage, packed, 
                n_item, capacity, weights, 
                heuristic_model_folder, input, train_n_item, train_capacity_ratio,
                train_lambdas_0, train_lambdas_1, 
                train_lambdas_2, train_lambdas_3,
                discrete_coeff, scaling_factor, seed, heuristic
            )

        elif heuristic == 'zero':
            return create_zero_heuristic_function()

        elif heuristic == 'greedy':
            return create_greedy_heuristic_function(
                n_item, weights, capacity, 
                means, deviations, skewnesses, kurtosis, lambdas,
                total_weight, stage, 
                heuristic_type=greedy_heuristic_type, 
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
                    dp_model_name, 
                    total_weight, stage, packed, 
                    n_item, capacity, weights, 
                    policy_model_folder, 
                    input, 
                    train_n_item, train_capacity_ratio,
                    train_lambdas_0, train_lambdas_1, 
                    train_lambdas_2, train_lambdas_3,
                    discrete_coeff, softmax_temperature, seed, 
                    rl_algorithm=policy_name)

    if policy_name == 'ppo' and heuristic == 'dual':
        _policy_f_evaluator = policy_f_evaluator_maximization
    elif policy_name == 'ppo' and heuristic != 'dual':
        _policy_f_evaluator = policy_f_evaluator
    else:
        _policy_f_evaluator = None

    solver_params = {
        "h_evaluator": h_evaluator if heuristic != 'dual' else None,
        "g_evaluators": g_evaluators if heuristic != 'dual' else None,
        "f_operator": dp.FOperator.Plus if heuristic != 'dual' else None,
        "policy": policy if policy_name == 'ppo' else None,
        "policy_f_evaluator": _policy_f_evaluator,
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
    start_clock_time=None,
):
    is_terminated = False

    while not is_terminated:
        solution, is_terminated = solver.search_next()

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
        investments = []

        for i, t in enumerate(solution.transitions):
            if t.name == "invest":
                print("invest {}".format(i))
                investments.append(i)

        print(" ".join(map(str, investments)))

        print("best bound: {}".format(solution.best_bound))
        print("cost: {}".format(solution.cost))

        if solution.is_optimal:
            print("optimal cost: {}".format(solution.cost))

        return investments, solution.cost, solution_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # DIDP parameters
    parser.add_argument("--model", default="narita", type=str) # DIDP model (narita)
    parser.add_argument("--solver-name", default="CABS", type=str) # CABS, ACPS, APPS, CAASDy
    parser.add_argument('--heuristic', type=str, default='dual') # dual, dqn, zero, greedy
    parser.add_argument("--policy-name", default="none", type=str) # none, ppo
    parser.add_argument("--greedy-heuristic-type", default=4, type=int) # 1, 2, 3, 4
    parser.add_argument("--time-out", default=1800, type=int) # seconds, -1 means no time limit
    parser.add_argument("--non-zero-base-case", action="store_true")
    parser.add_argument("--threads", default=1, type=int)
    parser.add_argument("--initial-beam-size", default=1, type=int) # initial beam size for CABS, ACPS, APPS

    # Test instance parameters
    parser.add_argument("--n-item", default=20, type=int) # problem size
    parser.add_argument("--lb", default=0, type=int) # lowest value allowed for generating the lambda values
    parser.add_argument("--ub", default=100, type=int) # highest value allowed for generating the lambda values
    parser.add_argument("--capacity-ratio", default=0.5, type=float) # capacity of the instance is capacity_ratio * (sum of all the item weights)
    parser.add_argument("--lambdas-0", default=1, type=int) # 1st moment factor
    parser.add_argument("--lambdas-1", default=5, type=int) # 2nd moment factor
    parser.add_argument("--lambdas-2", default=5, type=int) # 3rd moment factor
    parser.add_argument("--lambdas-3", default=5, type=int) # 4th moment factor
    parser.add_argument("--discrete-coeff", default=0, type=int) # 1 if the cost is discrete, 0 otherwise

    # 0: no dual bound, 1: sum of remaining items, 2: 1 and max efficiency of remaining items
    parser.add_argument("--dual-bound", default=2, type=int)

    parser.add_argument("--num-instance", default=100, type=int)
    parser.add_argument('--seed', type=int, default=0)

    # RL model parameters
    parser.add_argument('--scaling-factor', type=float, default=0.001) # scaling factor of the reward function
    parser.add_argument('--softmax-temperature', type=float, default=1.0) # temperature for ppo
    parser.add_argument('--no-policy-accumulation', default=0, type=int) # 0 if accumulated policy from the root node is used for guidance
    parser.add_argument('--train-n-item', type=int, default=20) # training instance size
    parser.add_argument('--train-lb', type=int, default=0)
    parser.add_argument('--train-ub', type=int, default=100)
    parser.add_argument('--train-capacity-ratio', type=float, default=0.5)
    parser.add_argument('--train-lambdas-0', type=int, default=1)
    parser.add_argument('--train-lambdas-1', type=int, default=5)
    parser.add_argument('--train-lambdas-2', type=int, default=5)
    parser.add_argument('--train-lambdas-3', type=int, default=5)
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

    dataset_path = "n%d/lb%d-ub%d-cr%s-lmd%d%d%d%d-ni%d-s%d" % \
                        (args.n_item, args.lb, args.ub,                   
                         str(args.capacity_ratio).replace('.', ''),
                         args.lambdas_0, args.lambdas_1, 
                         args.lambdas_2, args.lambdas_3,
                         args.num_instance, args.seed)
    load_folder = os.path.join("./instances", "portfolio", dataset_path)

    print("load_folder", load_folder)
    assert os.path.exists(load_folder), "The dataset does not exist."

    start = time.perf_counter()

    # Read Portfolio instance
    n, capacity, weights, m, d, s, k, lmd = \
        read(os.path.join(load_folder, args.file))

    # Define the DP model
    if args.model == "narita":
        model, total_weight, i, packed, means, deviations, skewnesses, kurtosis, lambdas = \
            create_DPmodel_narita(
                n, capacity, weights, m, d, s, k, lmd, 
                discrete_coeff=args.discrete_coeff,
                dual_bound=args.dual_bound
        )
    
    # DQN model folder
    heuristic_model_folder = "./rl_agent/hybrid_cp_rl_solver/selected-models/dqn/portfolio/%s/n-item-%d/capacity-ratio-%s/lambdas-%d-%d-%d-%d" % \
                    (args.model, args.train_n_item,
                        str(args.train_capacity_ratio),
                        args.train_lambdas_0, args.train_lambdas_1, 
                        args.train_lambdas_2, args.train_lambdas_3)
    policy_model_folder = "./rl_agent/hybrid_cp_rl_solver/selected-models/ppo/portfolio/%s/n-item-%d/capacity-ratio-%s/lambdas-%d-%d-%d-%d" % \
                    (args.model, args.train_n_item, 
                        str(args.train_capacity_ratio), 
                        args.train_lambdas_0, args.train_lambdas_1, 
                        args.train_lambdas_2, args.train_lambdas_3)

    solver = create_solver(
        # DIDP parameters
        model, total_weight, i, packed, 
        m, d, s, k, lmd,  
        args.n_item,
        capacity,
        weights,
        args.solver_name,
        args.discrete_coeff,
        greedy_heuristic_type=args.greedy_heuristic_type,
        initial_beam_size=args.initial_beam_size,
        time_limit=args.time_out,
        heuristic=args.heuristic,
        policy_name=args.policy_name,
        # RL parameters
        dp_model_name=args.model,
        scaling_factor=args.scaling_factor,
        heuristic_model_folder=heuristic_model_folder,
        policy_model_folder=policy_model_folder,
        softmax_temperature=args.softmax_temperature,
        input=os.path.join(load_folder, args.file),
        train_n_item=args.train_n_item,
        train_capacity_ratio=args.train_capacity_ratio,
        train_lambdas_0=args.train_lambdas_0,
        train_lambdas_1=args.train_lambdas_1,
        train_lambdas_2=args.train_lambdas_2,
        train_lambdas_3=args.train_lambdas_3,
        seed=args.train_seed,
        no_policy_accumulation=no_policy_accumulation,
        )

    investments, cost, summary = solve(
        solver,
        start_clock_time=start,
    )

    solution = [1 if i in investments else 0 for i in range(n)]
    if cost is not None and validate(args.n_item, capacity, weights, solution):
        print("The solution is valid.")
    else:
        print("The solution is invalid.")