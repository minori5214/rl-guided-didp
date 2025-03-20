#!/usr/bin/env python3

import argparse
import time
import os

import torch
torch.set_num_threads(1)
print("cpu thread used: ", torch.get_num_threads())
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np

import didppy as dp
import read_knapsack

from knapsack_agent import KnapsackAgent
from cp_rl_solver.problem.knapsack.environment.environment import Environment
from cp_rl_solver.problem.knapsack.environment.state import *

#start = time.perf_counter()

import sys

#h_val_dict = {}
#for i in range(50):
#    h_val_dict[i] = []

def create_model_cappart(n, capacity, weights, profits, 
                         float_cost=True, target=None, dual_bound=1):
    if target is None:
        target_stage = 0
        target_total_weight = 0
    else:
        assert type(target) == list
        assert len(target) == 2
        assert type(target[0]) == int
        assert isinstance(target[1], (int, float))
        target_stage = target[0]
        target_total_weight = target[1]

    model = dp.Model(maximize=True, float_cost=float_cost)

    item = model.add_object_type(number=n)
    if float_cost:
        total_weight = model.add_float_var(target=target_total_weight)
    else:
        total_weight = model.add_int_var(target=target_total_weight)

    i = model.add_element_var(object_type=item, target=target_stage)

    if float_cost:
        w = model.add_float_table(weights)
        p = model.add_float_table(profits)
    else:
        w = model.add_int_table(weights)
        p = model.add_int_table(profits)

    if float_cost == 1:
        _cost = dp.FloatExpr.state_cost()
    else:
        _cost = dp.IntExpr.state_cost()

    ignore = dp.Transition(
        name="ignore",
        cost=_cost,
        effects=[(i, i + 1)],
        preconditions=[i < n],
    )
    model.add_transition(ignore)

    pack = dp.Transition(
        name="pack",
        cost=p[i] + _cost,
        effects=[(total_weight, total_weight + w[i]), (i, i + 1)],
        preconditions=[i < n, total_weight + w[i] <= capacity],
    )
    model.add_transition(pack)

    if dual_bound >= 1:
        # Simply add all the profits of the remaining items
        # This can be precomputed as a table
        # If the weight exceeds the capacity, ignore the item (additional condition in if_then_else)
        #NOTE: A more efficient version of this dual bound is below.
        #      Update the code once the current set of experiments is done
        #     (e.g., when we move on to the integer profit instances)
        #rem_profit_table = [sum(profits[j:]) for j in range(n)] + [0.0]
        #rem_profit = model.add_float_table(rem_profit_table)
        #model.add_dual_bound(rem_profit[i])

        remaining = sum([(i <= j).if_then_else(p[j], 0) for j in range(n)])
        model.add_dual_bound(remaining)
    if dual_bound >= 2:
        # For each stage j, Max efficiency (j) * remaining capacity
        max_efficiency = [max(profits[j] / weights[j] for j in range(m, n)) 
                            for m in range(n)] + [0] # [0] is for the base case
        max_eff = model.add_float_table(max_efficiency)
        model.add_dual_bound(max_eff[i] * (capacity - total_weight))


    model.add_base_case([i == n])

    return model, total_weight, i

def create_model_kuroiwa(n, capacity, weights, profits, float_cost=True, target=None):
    if target is None:
        target_stage = 0
    else:
        assert type(target) == list
        assert len(target) == 1
        assert type(target[0]) == int
        target_stage = target[0]

    model = dp.Model(maximize=True, float_cost=float_cost)

    item = model.add_object_type(number=n)

    if float_cost:
        r = model.add_float_var(target=capacity)
    else:
        r = model.add_int_var(target=capacity)

    i = model.add_element_var(object_type=item, target=target_stage)

    if float_cost:
        w = model.add_float_table(weights)
        p = model.add_float_table(profits)
    else:
        w = model.add_int_table(weights)
        p = model.add_int_table(profits)

    if float_cost == 1:
        _cost = dp.FloatExpr.state_cost()
    else:
        _cost = dp.IntExpr.state_cost()

    pack = dp.Transition(
        name="pack",
        cost=p[i] + _cost,
        effects=[(r, r - w[i]), (i, i + 1)],
        preconditions=[i < n, r >= w[i]],
    )
    model.add_transition(pack)

    ignore = dp.Transition(
        name="ignore",
        cost=_cost,
        effects=[(i, i + 1)],
        preconditions=[i < n],
    )
    model.add_transition(ignore)

    model.add_base_case([i == n])

    return model, r, i

def create_model_narita(n, capacity, weights, profits, 
                         float_cost=True, target=None, dual_bound=1):
    # For the base case
    weights = weights + [0]
    profits = profits + [0]

    if target is None:
        target_unselected = [i for i in range(n+1)] # index n is for the base case
        target_total_weight = 0 
        target_last_item = n+1 # item chosen last time; n+1 is a dummy value
    else:
        assert type(target) == list
        assert len(target) == 3
        assert type(target[0]) == list
        assert isinstance(target[1], (int, float))
        assert isinstance(target[1], int)
        target_unselected = target[0]
        target_total_weight = target[1]
        target_last_item = target[2]

    model = dp.Model(maximize=True, float_cost=float_cost)

    item = model.add_object_type(number=n+1)
    unselected = model.add_set_var(object_type=item, target=target_unselected)
    if float_cost:
        total_weight = model.add_float_var(target=target_total_weight)
    else:
        total_weight = model.add_int_var(target=target_total_weight)

    last_item = model.add_element_var(object_type=item, target=target_last_item)

    if float_cost:
        w = model.add_float_table(weights)
        p = model.add_float_table(profits)
    else:
        w = model.add_int_table(weights)
        p = model.add_int_table(profits)

    if float_cost == 1:
        state_cost = dp.FloatExpr.state_cost()
    else:
        state_cost = dp.IntExpr.state_cost()
    name_to_item = {}

    for i in range(0, n+1):
        name = "pack {}".format(i)
        name_to_item[name] = i
        pack = dp.Transition(
            name=name,
            cost=p[i] + state_cost,
            effects=[
                (unselected, unselected.remove(i)),
                (total_weight, total_weight + w[i]),
                (last_item, i),
            ],
            preconditions=[unselected.contains(i), last_item != n, total_weight + w[i] <= capacity],
        )
        model.add_transition(pack)

    if dual_bound >= 1:
        # Simply add all the profits of the remaining items
        remaining = p[unselected]
        model.add_dual_bound(remaining)
    if dual_bound >= 2:
        # For each stage j, Max efficiency (i) * remaining capacity
        ratio = model.add_float_table([profits[i]/(weights[i] + 1e-6) for i in range(n+1)])
        max_efficiency = ratio.max(unselected)
        model.add_dual_bound(max_efficiency * (capacity - total_weight))

    model.add_base_case([last_item == n])

    return model, name_to_item, unselected, total_weight, last_item

def create_dummy_policy(action_size):

    def policy(state):
        return np.zeros(action_size)

    return policy

def create_RL_policy(total_weight, stage,
                    n_item, capacity, weights, 
                    model_folder, input, train_n_item, train_capacity_ratio,
                    train_cor_type, train_sort, softmax_temperature, seed, 
                    rl_algorithm='ppo'):
    # PPO agent
    agent = KnapsackAgent(model_folder, input, train_n_item, train_capacity_ratio,
                 train_cor_type, train_sort, None, seed, rl_algorithm)

    # TSPTW environment
    env = Environment(agent.instance, 1)

    def to_RLstate(DIDPstate):
        weight = DIDPstate[total_weight]
        cur_stage = DIDPstate[stage]
        #print("State", weight, cur_stage, capacity)

        # Not possible to insert the item if its insertion exceed the portfolio capacity
        if cur_stage == n_item:
            available_action = set()
        elif weight + weights[cur_stage] > capacity:
            available_action = set([0])
        else:
            available_action = set([0, 1])
        
        # State.tour can be left empty
        return State(agent.instance, weight, cur_stage, available_action)

    def policy(state):
        #print("state", state[total_weight], state[stage])
        # Convert state to RL state
        RLstate = to_RLstate(state)

        state_feats = env.make_nn_input(RLstate, "cpu")
        #print("state_feats", state_feats)
        avail = env.get_valid_actions(RLstate)

        available_tensor = torch.FloatTensor(avail)
        #print("avail", avail)

        with torch.no_grad():
            batched_set = state_feats.unsqueeze(0)
            out = agent.model(batched_set)
            action_probs = out.squeeze(0)
        #print("action_probs 0", action_probs)

        action_probs = action_probs + torch.abs(torch.min(action_probs))
        action_probs = action_probs - torch.max(action_probs * available_tensor)
        action_probs = agent.actor_critic_network.masked_softmax2(
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
        #print("probabilities", probabilities)

        # Policy must return log probabilities
        log_probabilities = np.log(probabilities)
        #print("log_probabilities", log_probabilities)

        return log_probabilities

    return policy

def policy_f_evaluator(log_pi, g, h):
    if h is None:
        h =0.0 # if there is no dual bound, only g is used

    # PHS priority function in logarithm
    #f = np.log(1) - log_pi
    if (g + h) == 0:
        return -log_pi

    #print("g", g)
    #print("h", h)
    #print("log_pi", log_pi)
    f = -(np.log(g + h) + log_pi)
    #print("f", f)
    return f

def policy_f_evaluator_noeta(log_pi, g, h):
    if g == 0:
        return -log_pi

    f = -(np.log(g) + log_pi)
    return f

def policy_f_evaluator_noeta_nog(log_pi, g, h):
    f = -log_pi
    return f

def policy_f_evaluator_noeta_nopi(log_pi, g, h):
    return -np.log(g)

def policy_f_evaluator_for_negative(log_pi, g, h):
    # If g- and h-values are be negative, we take the negative to make it positive.
    # Then, it becomes maximization, so we multiply pi (or add log pi to the logarithm)
    # and take the negative to put it back to minimization
    f = -(np.log(-(g + h)) + log_pi)
    return f

def create_RL_g_function(profits, stage, is_pack, scaling_factor):
    def g_function(parent_g, state):
        cost = is_pack * profits[state[stage]]

        # Scale the cost
        reward =  cost * scaling_factor

        # Maximization problem
        g = parent_g + (-reward)
        #print("stage={}, g={}, cost={}, scale={}".format(state[stage], g, cost, scaling_factor))

        return g

    return g_function

def create_RL_heuristic_function_cappart(
    model_folder, input, n_item, weights, profits, capacity,
    train_n_item, train_capacity_ratio,
    train_cor_type, train_sort, gnn,
    total_weight, stage,
    seed
):
    # DQN agent
    print("model_folder: ", model_folder)
    print("input: ", input)
    print("train_n_item: ", train_n_item)
    print("train_capacity_ratio: ", train_capacity_ratio)
    print("train_cor_type: ", train_cor_type)
    print("train_sort: ", train_sort)
    print("gnn: ", gnn)
    agent = KnapsackAgent(model_folder, input, train_n_item, train_capacity_ratio,
                 train_cor_type, train_sort, gnn, seed, 'dqn')

    # TSPTW environment
    env = Environment(agent.instance, 1)

    def to_RLstate(DIDPstate):
        cur_weight = DIDPstate[total_weight]
        cur_stage = DIDPstate[stage]

        # Not possible to insert the item if its insertion exceed the portfolio capacity
        if cur_stage == n_item:
            available_action = set()
        elif cur_weight + weights[cur_stage] > capacity:
            available_action = set([0])
        else:
            available_action = set([0, 1])
        
        # State.tour can be left empty
        return State(agent.instance, cur_weight, cur_stage, available_action)

    def RL_heuristic(state):
        # Convert state to RL state
        RLstate = to_RLstate(state)

        #print("weights={}, profits={}, capacity={}".format(
        #    weights, profits, capacity))
        #print("Cur state: cur_weight={}, stage={}".format(
        #    RLstate.weight, RLstate.stage))

        # Available actions
        avail = env.get_valid_actions(RLstate)
        available = avail.astype(bool)

        # If no available action, it is the base case - calculate the true value
        if np.any(available) == False:
            return 0.0

        nn_input = env.make_nn_input(RLstate, 'cpu')
        nn_input = nn_input.unsqueeze(0)

        # Get RL prediction
        with torch.no_grad():
            res = agent.model(nn_input)

        out = res.cpu().numpy().squeeze(0)

        # Mask unavailable actions
        v = np.max(out[available])

        h = -v
        #print("stage={}, h={}".format(state[stage], h))
        return h

    return RL_heuristic


def create_zero_g_function(profits, stage, is_pack):
    def g_function(parent_g, state):
        cost = is_pack * profits[state[stage]]
        g = parent_g - cost # Maximization problem

        assert parent_g >= g
        #print("stage={}, g={}, cost={}, is_pack={}".format(
        #    state[stage], g, cost, is_pack))

        return g

    return g_function


def create_zero_heuristic_function():
    def zero_heuristic(state):
        return 0

    return zero_heuristic

def create_greedy_heuristic_function(n, weights, profits, capacity, 
                                     total_weight, stage, item_order='best_ratio', debug=False):
    # item_oredr (str): 'best_ratio' or 'random'

    #max_efficiency = [max(profits[j] / weights[j] for j in range(m, n)) 
    #                        for m in range(n)] + [0] # [0] is for the base case

    def greedy_heuristic_bestratio(state):
        # Knapsack: Choose the item with best ratio

        cur_weight = state[total_weight]
        cur_stage = state[stage]

        _cur_weight = cur_weight
        _cur_stage = cur_stage

        solution = []
        h = 0

        # Consider each item in the remaining items (_cur_stage to n), 
        # and choose the best ratio item until the capacity is full
        for i in range(_cur_stage, n):
            if _cur_weight + weights[i] <= capacity:
                solution.append(i)
                _cur_weight += weights[i]
                h += profits[i]
        
        assert _cur_weight <= capacity, "The solution exceeds the capacity"
        assert len(solution) == len(set(solution)), "The solution contains duplicate items. solution={}".format(solution)
        assert h == sum([profits[j] for j in solution]), "The solution cost is incorrect"
        
        if debug:
            # Make sure that max efficiency item is always chosen at each iteration
            # Assume that the items are sorted by the best ratio
            for i in solution:
                remaining_items = list(range(i, n))
                efficiency = [profits[j] / weights[j] for j in remaining_items]
                assert profits[i] / weights[i] == max(efficiency)

        return -h # maximization

    if item_order == 'best_ratio': return greedy_heuristic_bestratio
    else: raise NotImplementedError


def create_dual_heuristic_function(
        n, weights, profits, capacity, 
        total_weight, stage, dual_bound=2
        ):
    #NOTE: This is used only for debugging; for actual experiments, use regular DIDP features
    def dual_heuristic(state):
        if dual_bound == 0:
            return 0


        cur_weight = state[total_weight]
        cur_stage = state[stage]

        dual_bound_1 = 0
        dual_bound_2 = 0
        
        if dual_bound >= 1:
            dual_bound_1 = sum([profits[j] for j in range(cur_stage, n)])

        if dual_bound >= 2:
            max_efficiency = [max(profits[j] / weights[j] for j in range(m, n)) 
                            for m in range(n)] + [0] # [0] is for the base case
            dual_bound_2 = max_efficiency[cur_stage] * (capacity - cur_weight)

        h = -(min(dual_bound_1, dual_bound_2))

        print("dual 1={}, dual 2={}, h={}, rem_cap={}".format(
            dual_bound_1, dual_bound_2, h, capacity - cur_weight))

        return h

    return dual_heuristic

def create_solver(
    model, 
    total_weight, 
    stage, 
    unselected,
    last_item,
    n_item,
    capacity,
    weights,
    profits,
    model_name,
    solver_name,
    scaling_factor=0.0001,
    policy_name=None,
    initial_beam_size=1,
    time_limit=None,
    expansion_limit=None,
    max_beam_size=None,
    # RL heuristic
    heuristic='dual',
    heuristic_model_folder=None,
    policy_model_folder=None,
    softmax_temperature=1.0,
    input=None,
    train_n_item=None,
    train_capacity_ratio=None,
    train_cor_type='strongly', 
    train_sort=1, 
    gnn='knapsacktanh',
    seed=None,
    no_policy_accumulation=True,
    nouse_eta=1,
    nouse_g=1
):
    if heuristic == 'dqn':
        assert heuristic_model_folder is not None
        assert input is not None
        assert train_n_item is not None
        assert train_capacity_ratio is not None
        assert seed is not None

    if heuristic != 'dual' and policy_name == 'ppo':
        raise NotImplementedError
    elif heuristic != 'dual':
        if heuristic == 'dqn':
            g_evaluators = {}
            g_evaluators["pack"] = create_RL_g_function(
                    profits, stage, 1, scaling_factor
            )

            g_evaluators["ignore"] = create_RL_g_function(
                profits, stage, 0, scaling_factor
            )

            h_evaluator = create_RL_heuristic_function_cappart(
                heuristic_model_folder, input, n_item, weights, profits, capacity,
                train_n_item, train_capacity_ratio,
                train_cor_type, train_sort, gnn,
                total_weight, stage,
                seed
            )
        elif heuristic == 'zero':
            g_evaluators = {}
            g_evaluators["pack"] = create_zero_g_function(
                    profits, stage, 1
            )

            g_evaluators["ignore"] = create_zero_g_function(
                profits, stage, 0
            )

            h_evaluator = create_zero_heuristic_function()
        elif heuristic == 'greedy':
            g_evaluators = {}
            g_evaluators["pack"] = create_zero_g_function(
                    profits, stage, 1
            )

            g_evaluators["ignore"] = create_zero_g_function(
                profits, stage, 0
            )

            h_evaluator = create_greedy_heuristic_function(
                n_item, weights, profits, capacity, 
                total_weight, stage, debug=False
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
                time_limit=time_limit,
                quiet=False,
                no_policy_accumulation=no_policy_accumulation,
                expansion_limit=expansion_limit,
                max_beam_size=max_beam_size
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
        else:
            raise NotImplementedError
    elif policy_name == 'ppo':
        policy = create_RL_policy(total_weight, stage,
                    n_item, capacity, weights, 
                    policy_model_folder, input, train_n_item, train_capacity_ratio,
                    train_cor_type, train_sort, softmax_temperature, seed, 
                    rl_algorithm=policy_name)

        if nouse_eta == 1 and nouse_g == 1:
            _policy_f_evaluator = policy_f_evaluator_noeta_nog
        elif nouse_eta == 1:
            _policy_f_evaluator = policy_f_evaluator_noeta
        elif nouse_eta == 0 and nouse_g == 0:
            _policy_f_evaluator = policy_f_evaluator
        else:
            raise NotImplementedError

        if solver_name == "CAASDy":
            solver = dp.UserPriorityCAASDy(
                model,
                policy=policy,
                policy_f_evaluator=_policy_f_evaluator,
                time_limit=time_limit,
                quiet=False,
                no_policy_accumulation=no_policy_accumulation,
                expansion_limit=expansion_limit
            )
        elif solver_name == "CABS":
            solver = dp.UserPriorityCABS(
                model,
                policy=policy,
                policy_f_evaluator=_policy_f_evaluator,
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
                policy_f_evaluator=_policy_f_evaluator,
                time_limit=time_limit,
                quiet=False,
                no_policy_accumulation=no_policy_accumulation,
                expansion_limit=expansion_limit
            )
        elif solver_name == "APPS":
            solver = dp.UserPriorityAPPS(
                model,
                policy=policy,
                policy_f_evaluator=_policy_f_evaluator,
                time_limit=time_limit,
                quiet=False,
                no_policy_accumulation=no_policy_accumulation,
                expansion_limit=expansion_limit
            )
        else:
            raise NotImplementedError
    elif heuristic == 'dual' and nouse_eta == 1:
        policy = create_dummy_policy(action_size=2)

        if solver_name == "CAASDy":
            solver = dp.UserPriorityCAASDy(
                model,
                policy=policy,
                policy_f_evaluator=policy_f_evaluator_noeta_nopi,
                time_limit=time_limit,
                quiet=False,
                no_policy_accumulation=no_policy_accumulation,
                expansion_limit=expansion_limit
            )
        elif solver_name == "CABS":
            solver = dp.UserPriorityCABS(
                model,
                policy=policy,
                policy_f_evaluator=policy_f_evaluator_noeta_nopi,
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
                policy_f_evaluator=policy_f_evaluator_noeta_nopi,
                time_limit=time_limit,
                quiet=False,
                no_policy_accumulation=no_policy_accumulation,
                expansion_limit=expansion_limit
            )
        elif solver_name == "APPS":
            solver = dp.UserPriorityAPPS(
                model,
                policy=policy,
                policy_f_evaluator=policy_f_evaluator_noeta_nopi,
                time_limit=time_limit,
                quiet=False,
                no_policy_accumulation=no_policy_accumulation,
                expansion_limit=expansion_limit
            )
        else:
            raise NotImplementedError

    elif solver_name == "CAASDy":
        solver = dp.CAASDy(
            model,
            time_limit=time_limit,
            quiet=False,
            expansion_limit=expansion_limit
        )
    elif solver_name == "CABS":
        # Debug
        #g_evaluators = {}
        #g_evaluators["pack"] = create_zero_g_function(
        #        profits, stage, 1
        #)

        #g_evaluators["ignore"] = create_zero_g_function(
        #    profits, stage, 0
        #)

        #h_evaluator = create_dual_heuristic_function(
        #            n_item, weights, profits, capacity, 
        #            total_weight, stage, dual_bound=2
        #)

        #solver = dp.UserPriorityCABS(
        #    model,
        #    h_evaluator=h_evaluator,
        #    g_evaluators=g_evaluators,
        #    f_operator=dp.FOperator.Plus,  # g + h
        #    initial_beam_size=initial_beam_size,
        #    time_limit=time_limit,
        #    quiet=False,
        #    no_policy_accumulation=no_policy_accumulation,
        #    expansion_limit=expansion_limit,
        #    max_beam_size=max_beam_size
        #)

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
    n, 
    model_name,
    solver,
    name_to_item,
    history,
    start_clock_time=None,
    save=1
):
    if start_clock_time is None:
        start_clock_time = time.perf_counter()
    if save:
        with open(history, "w") as f:
            f.write("time, solve_time, expanded, generated, cost, best_bound, is_optimal\n")

            is_terminated = False

            while not is_terminated:
                solution, is_terminated = solver.search_next()

                if model_name in ["cappart", "kuroiwa"]:
                    _solution = []
                    for i, t in enumerate(solution.transitions):
                        if t.name == "pack":
                            _solution.append(1)
                        else:
                            _solution.append(0)
                elif model_name == "narita":
                    _packed = []
                    for t in solution.transitions:
                        _packed.append(name_to_item[t.name])
                    _packed.pop(-1) # base case
                    _solution = [1 if i in _packed else 0 for i in range(n)]

                if solution.cost is not None:
                    f.write(
                        "{}, {}, {}, {}, {}, {}, {}\n".format(
                            time.perf_counter() - start_clock_time, 
                            solution.time,
                            solution.expanded,
                            solution.generated,
                            sum(profit_list[j] * _solution[j] for j in range(n)), # cost
                            solution.best_bound,
                            solution.is_optimal)
                    )
                    f.flush()
    else:
        is_terminated = False

        while not is_terminated:
            solution, is_terminated = solver.search_next()

            _solution = []
            for i, t in enumerate(solution.transitions):
                if t.name == "pack":
                    _solution.append(1)
                else:
                    _solution.append(0)


        """
        while not is_terminated:
            try:
                solution, is_terminated = solver.search_next()

                if solution.cost is not None:
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
            except:
                print("Search halted. Memory Error?")
                break
        """

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
        if model_name in ["cappart", "kuroiwa"]:
            packed = []

            for i, t in enumerate(solution.transitions):
                if t.name == "pack":
                    print("pack {}".format(i))
                    packed.append(i)
        elif model_name == "narita":
            packed = []

            for t in solution.transitions:
                packed.append(name_to_item[t.name])
            packed.pop(-1) # base case

        print("best bound: {}".format(solution.best_bound))
        print("cost: {}".format(solution.cost))

        if solution.is_optimal:
            print("optimal cost: {}".format(solution.cost))

        return packed, solution.cost, solution_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    # DIDP parameters
    parser.add_argument("--policy-name", default="", type=str)
    parser.add_argument("--time-out", default=3600, type=int) # -1 means no time limit
    parser.add_argument("--expansion-limit", default=-1, type=int) # -1 means no node expansion limit
    parser.add_argument("--solver-name", default="CABS", type=str)
    parser.add_argument("--threads", default=1, type=int)
    parser.add_argument("--initial-beam-size", default=1, type=int)
    parser.add_argument("--parallel-type", default=0, type=int)
    parser.add_argument("--float-cost", default=1, type=int)
    parser.add_argument('--heuristic', type=str, default='dual') # dual, dqn, zero
    parser.add_argument('--softmax-temperature', type=float, default=1.0)
    parser.add_argument('--no-policy-accumulation', default=1, type=int)

    # Test instance parameters
    parser.add_argument("--model", default="cappart", type=str)
    parser.add_argument("--n-item", default=50, type=int)
    parser.add_argument("--lb", default=0, type=int)
    parser.add_argument("--ub", default=10000, type=int)
    parser.add_argument("--capacity-ratio", default=0.5, type=float)
    parser.add_argument("--cor-type", default="strongly", type=str)
    parser.add_argument("--sort", default=1, type=int) # whether the instance is sorted by best-ratio
    parser.add_argument("--gnn", default="knapsacktanh", type=str) # knapsacktanh, settransformer

    parser.add_argument("--num-instance", default=100, type=int)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument("--dual-bound", default=0, type=int)

    # RL model parameters
    parser.add_argument('--train-n-item', type=int, default=50)
    parser.add_argument('--train-capacity-ratio', type=float, default=0.5)
    parser.add_argument("--train-cor-type", default="strongly", type=str)
    parser.add_argument("--train_sort", default=1, type=int) # whether the instance is sorted by best-ratio
    parser.add_argument('--train-seed', type=int, default=0)
    parser.add_argument("--train-reward-scaling", default=0.0001, type=float)

    parser.add_argument('--file', type=str, default="0.txt")
    parser.add_argument('--save', type=str, default=1)

    parser.add_argument('--nouse-eta', type=int, default=0) # 1 if dual is not used in policy f-evaluator
    parser.add_argument('--nouse-g', type=int, default=0) # 1 if g is not used in policy f-evaluator

    parser.add_argument('--max-beam-size', type=int, default=-1) # for CABS

    args = parser.parse_args()
    if args.time_out == -1:
        args.time_out = None
    if args.expansion_limit == -1:
        args.expansion_limit = None
    if args.max_beam_size == -1:
        args.max_beam_size = None

    if args.heuristic not in ['dual', 'dqn', 'zero', 'greedy']:
        raise NotImplementedError

    print("Instance: ", args.file)

    dataset_path = "n%d/lb%d-ub%d-cr%s-corr%s-sort%d-ni%d-s%d/" % \
                        (args.n_item, args.lb, args.ub,                   
                         str(args.capacity_ratio).replace('.', ''),
                         args.cor_type, args.sort,
                         args.num_instance, args.seed)

    load_folder = os.path.join("./instances", "knapsack", dataset_path)
    history_folder = os.path.join(
                        "./history", 
                        "exlim%d" % args.expansion_limit if args.expansion_limit is not None else "no-exlim",
                        args.model, 
                        args.solver_name, 
                        "h-%s-%s" % (args.heuristic, args.policy_name), 
                        'knapsack',
                        "db%d%s%s" % (args.dual_bound, 
                                    "-eta0" if args.nouse_eta == 1 else "",
                                    "-g0" if args.nouse_g == 1 else ""),
                        dataset_path
                        )
    
    print(load_folder)
    assert os.path.exists(load_folder), "The dataset does not exist."
    if not os.path.exists(history_folder):
        os.makedirs(history_folder, exist_ok=True)
    
    assert args.n_item == args.train_n_item, \
        "train instance size (n={}) must be equal to test size (n={})".format(
            args.train_n_item, args.n_item)

    #log_file_path = os.path.join(history_folder, 'output.log')
    #log_file = open(log_file_path, 'w')
    #sys.stdout = log_file

    start = time.perf_counter()

    n, capacity, weight_list, profit_list = read_knapsack.read(os.path.join(load_folder, args.file))
    print("capacity", capacity)
    print("weights", weight_list)
    print("profits", profit_list)

    if args.model == "cappart":
        model, total_weight, i = \
            create_model_cappart(
                n, capacity, weight_list, profit_list, float_cost=bool(args.float_cost),
                dual_bound=args.dual_bound
        )
        name_to_item = None
        unselected = None
        last_item = None
    elif args.model == "narita":
        model, name_to_item, unselected, total_weight, last_item = \
            create_model_narita(
                n, capacity, weight_list, profit_list, float_cost=bool(args.float_cost)
            )
        i = None
    else:
        raise NotImplementedError

    heuristic_model_folder = "./cp_rl_solver/selected-models/dqn/knapsack/%s/correlation-%s/n-item-%d/gnn-%s-sort-%d/" % \
                    (args.model,
                        args.train_cor_type, 
                        args.train_n_item,
                        args.gnn,
                        args.train_sort)

    policy_model_folder = "./cp_rl_solver/selected-models/ppo/knapsack/%s/correlation-%s/n-item-%d/sort-%d/" % \
                    (args.model,
                        args.train_cor_type, 
                        args.train_n_item,
                        args.train_sort)

    no_policy_accumulation = args.no_policy_accumulation == 1

    solver = create_solver(
        model, total_weight, i, 
        unselected, 
        last_item,
        args.n_item,
        capacity,
        weight_list,
        profit_list,
        args.model,
        args.solver_name,
        scaling_factor=args.train_reward_scaling,
        policy_name=args.policy_name,
        initial_beam_size=args.initial_beam_size,
        time_limit=args.time_out,
        expansion_limit=args.expansion_limit,
        max_beam_size=args.max_beam_size,
        # RL heuristic
        heuristic=args.heuristic,
        heuristic_model_folder=heuristic_model_folder,
        policy_model_folder=policy_model_folder,
        softmax_temperature=args.softmax_temperature,
        input=os.path.join(load_folder, args.file),
        train_n_item=args.train_n_item,
        train_capacity_ratio=args.train_capacity_ratio,
        train_cor_type=args.train_cor_type,
        train_sort=args.train_sort,
        gnn=args.gnn,
        seed=args.train_seed,
        no_policy_accumulation=no_policy_accumulation,
        nouse_eta=args.nouse_eta,
        nouse_g=args.nouse_g
    )

    packed, cost, summary = solve(
        n, 
        args.model,
        solver,
        name_to_item,
        os.path.join(history_folder, "{}.csv".format(args.file.split('.')[0])),
        start_clock_time=start,
        save=args.save
    )

    solution = [1 if i in packed else 0 for i in range(n)]
    if cost is not None and \
        read_knapsack.validate(args.n_item, capacity, weight_list, profit_list, solution, cost):
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

    #for k, h in h_val_dict.items():
    #    if len(h) > 0:
    #        print("Stage {}: mean={}, std={}, len={}".format(k, np.mean(h), np.std(h), len(h)))
    
    #log_file.close()
