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
import read_portfolio

from portfolio_agent import PortfolioAgent
from cp_rl_solver.problem.portfolio.environment.environment import Environment
from cp_rl_solver.problem.portfolio.environment.state import *

#start = time.perf_counter()

import sys

def create_model_narita(n, capacity, w, m, d, s, k, lmd, discrete_coeff=0, target=None, 
                        dual_bound=0):
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
    moment_factors = model.add_int_table(lmd)

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
        cost=moment_factors[0]*(means[packed] + means[i])-\
                moment_factors[1]*((deviations[packed] + deviations[i])**(1/2))+\
                moment_factors[2]*((skewnesses[packed] + skewnesses[i])**(1/3))-\
                moment_factors[3]*((kurtosis[packed] + kurtosis[i])**(1/4)),
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
        # Sum of remaining items considering only the positive terms
        #NOTE: A more efficient version of this dual bound is below.
        #      Update the code once the current set of experiments is done
        #     (e.g., when we move on to the integer profit instances)
        #rem_m_table = [lmd[0]*sum(m[j:]) for j in range(n)] + [0.0]
        #rem_s_table = [lmd[2]*sum(s[j:])**(1/3) for j in range(n)] + [0.0]
        #rem_m = model.add_float_table(rem_m_table)
        #rem_s = model.add_float_table(rem_s_table)
        #model.add_dual_bound(rem_m[i] + rem_s[i])

        remaining_m = moment_factors[0]*sum([(i <= j).if_then_else(means[j], 0) for j in range(n)])
        remaining_s = moment_factors[2]*(sum([(i <= j).if_then_else(skewnesses[j], 0) for j in range(n)])**(1/3))
        model.add_dual_bound(remaining_m + remaining_s)

    if dual_bound >= 2:
        # Max efficiency of the remaining items considering only the positive terms
        # For each stage j, Max efficiency (j) * remaining capacity
        max_efficiency = [max((lmd[0]*m[j] + lmd[2]*s[j]**(1/3)) \
                              / (w[j] + 1e-6) for j in range(q, n)) 
                            for q in range(n)] + [0] # [0] is for the base case
        print("max_efficiency (model)", max_efficiency)
        max_eff = model.add_float_table(max_efficiency)
        model.add_dual_bound(max_eff[i] * (capacity - total_weight))

    model.add_base_case([i == n])

    return model, total_weight, i, packed, means, deviations, skewnesses, kurtosis, moment_factors

def create_model_cappart(n, capacity, w, m, d, s, k, lmd, discrete_coeff=0, target=None):
    if target is None:
        target_total_cost = 0
        target_stage = 0
        target_invested = [0 for _ in range(n)]
    else:
        assert type(target) == list
        assert len(target) == 3
        assert type(target[0]) == int or type(target[0]) == float
        assert type(target[1]) == int
        assert type(target[2]) == list
        target_total_cost = target[0]
        target_stage = target[1]
        target_invested = target[2]

    if discrete_coeff == 1:
        model = dp.Model(maximize=True)
    else:
        model = dp.Model(maximize=True, float_cost=True)

    investment = model.add_object_type(number=n)

    # c (total cost)
    if discrete_coeff == 1:
        total_cost = model.add_int_var(target=target_total_cost)
    else:
        total_cost = model.add_float_var(target=target_total_cost)

    # i (current item)
    i = model.add_element_var(object_type=investment, target=target_stage)

    # x (whether to take the item)
    x = {}
    for j in range(n):
        x[j] = model.add_int_var(target=target_invested[j])

    weights = model.add_int_table(w)
    means = model.add_int_table(m)
    deviations = model.add_int_table(d)
    skewnesses = model.add_int_table(s)
    kurtosis = model.add_int_table(k)
    moment_factors = model.add_int_table(lmd)

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
        cost= _cost,
        effects=[
            (total_cost, total_cost + weights[i]),
            (i, i + 1),
            ] + [(x[j], (i == j).if_then_else(1, x[j])) for j in range(n)],
        preconditions=[
            i < n,
            total_cost + weights[i] <= capacity
            ],
    )
    model.add_transition(invest)

    cost = moment_factors[0]*sum([means[j]*x[j] for j in range(n)])-\
        moment_factors[1]*(sum([deviations[j]*x[j] for j in range(n)])**(1/2))+\
        moment_factors[2]*(sum([skewnesses[j]*x[j] for j in range(n)])**(1/3))-\
        moment_factors[3]*(sum([kurtosis[j]*x[j] for j in range(n)])**(1/4))

    if discrete_coeff == 1:
        cost = int(cost)
    model.add_base_case([i == n], cost=cost)

    return model, total_cost, i, x, means, deviations, skewnesses, kurtosis, moment_factors

def create_dummy_policy(action_size):

    def policy(state):
        return np.zeros(action_size)

    return policy

def create_RL_policy(model, total_weight, stage, x, packed, 
                        n_item, capacity, weights, 
                        model_folder, input, train_n_item, train_capacity_ratio,
                        train_moment_factors_0, train_moment_factors_1, 
                        train_moment_factors_2, train_moment_factors_3,
                        discrete_coeff, softmax_temperature, seed, 
                        rl_algorithm='ppo'):
    # PPO agent
    agent = PortfolioAgent(model_folder, input, train_n_item, train_capacity_ratio,
                 train_moment_factors_0, train_moment_factors_1, 
                 train_moment_factors_2, train_moment_factors_3,
                 discrete_coeff, seed, rl_algorithm)

    # TSPTW environment
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
        
        if model == 'cappart':
            items_taken = [DIDPstate[x[j]] for j in range(agent.n_item)]
        elif model == 'narita':
            items_taken = [1 for j in range(agent.n_item) if j in DIDPstate[packed]]
        else:
            raise NotImplementedError

        # State.tour can be left empty
        return State(agent.instance, weight, cur_stage, available_action, items_taken)

    def policy(state):
        #print("state", state[total_weight], state[stage])
        # Convert state to RL state
        RLstate = to_RLstate(state)

        state_feats = env.make_nn_input(RLstate, "cpu")
        avail = env.get_valid_actions(RLstate)

        available_tensor = torch.FloatTensor(avail)

        with torch.no_grad():
            batched_set = state_feats.unsqueeze(0)
            out = agent.model(batched_set)
            action_probs = out.squeeze(0)
            #print("action_probs", action_probs)

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
    if h is None:
        h =0.0 # if there is no dual bound, only g is used

    # PHS priority function in logarithm
    #f = np.log(1) - log_pi
    if g == 0:
        return -log_pi

    f = -(np.log(g) + log_pi) # h is not used
    return f

def policy_f_evaluator_noeta_nog(log_pi, g, h):
    f = -log_pi # h is not used
    return f

def policy_f_evaluator_noeta_nopi(log_pi, g, h):
    return -np.log(g)

def policy_f_evaluator_for_negative(log_pi, g, h):
    # If g- and h-values are be negative, we take the negative to make it positive.
    # Then, it becomes maximization, so we multiply pi (or add log pi to the logarithm)
    # and take the negative to put it back to minimization
    f = -(np.log(-(g + h)) + log_pi)
    return f


def create_RL_g_function(scaling_factor, total_weight, stage, is_pack, x, packed, 
                         means, deviations, skewnesses, kurtosis, moment_factors,
                         discrete_coeff=0, model_name='cappart'):
    def g_function(parent_g, state):
        #print("g func: stage={}, cur_weight={}".format(state[stage], state[total_weight]))
        #print("items taken: ", state[x[0]], state[x[1]], state[x[2]], state[x[3]], state[x[4]], state[x[5]], state[x[6]], state[x[7]], state[x[8]], state[x[9]], state[x[10]], state[x[11]], state[x[12]], state[x[13]], state[x[14]], state[x[15]], state[x[16]], state[x[17]], state[x[18]], state[x[19]])
        #print("")
        if model_name == 'cappart':
            if discrete_coeff == 1:
                cost = 0
            else:
                cost = 0.0

            # Add the negative cost and scale it
            reward = cost * scaling_factor

            # Maximization problem
            g = parent_g + (-reward)

        elif model_name == 'narita':
            g = moment_factors[0]*(sum([means[j] for j in state[packed]])+is_pack*means[state[stage]])-\
                    moment_factors[1]*((sum([deviations[j] for j in state[packed]])+is_pack*deviations[state[stage]])**(1/2))+\
                    moment_factors[2]*((sum([skewnesses[j] for j in state[packed]])+is_pack*skewnesses[state[stage]])**(1/3))-\
                    moment_factors[3]*((sum([kurtosis[j] for j in state[packed]])+is_pack*kurtosis[state[stage]])**(1/4))
            #print("stage {}: packed={} raw g={}".format(state[stage], state[packed], g))
            g *= scaling_factor
            g = -g # Maximization problem
            #print("scaled g", g)
        else:
            raise NotImplementedError

        return g

    return g_function

def create_RL_heuristic_function(
    model, total_weight, stage, x, packed, means, deviations, skewnesses, kurtosis, moment_factors,
    n_item, capacity, weights, 
    model_folder, input, train_n_item, train_capacity_ratio,
    train_moment_factors_0, train_moment_factors_1, 
    train_moment_factors_2, train_moment_factors_3,
    discrete_coeff, scaling_factor, seed, rl_algorithm
):
    # DQN agent
    agent = PortfolioAgent(model_folder, input, train_n_item, train_capacity_ratio,
                 train_moment_factors_0, train_moment_factors_1, 
                 train_moment_factors_2, train_moment_factors_3,
                 discrete_coeff, seed, rl_algorithm)

    # TSPTW environment
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
        
        if model == 'cappart':
            items_taken = [DIDPstate[x[j]] for j in range(agent.n_item)]
        elif model == 'narita':
            items_taken = [1 for j in range(agent.n_item) if j in DIDPstate[packed]]
        else:
            raise NotImplementedError

        # State.tour can be left empty
        return State(agent.instance, weight, cur_stage, available_action, items_taken)

    def RL_heuristic(state):
        #print("h func: stage={}, cur_weight={}".format(state[stage], state[total_weight]))
        #print("items taken: ", state[x[0]], state[x[1]], state[x[2]], state[x[3]], state[x[4]], state[x[5]], state[x[6]], state[x[7]], state[x[8]], state[x[9]], state[x[10]], state[x[11]], state[x[12]], state[x[13]], state[x[14]], state[x[15]], state[x[16]], state[x[17]], state[x[18]], state[x[19]])


        # Convert state to RL state
        RLstate = to_RLstate(state)
        #print("RLstate: ", RLstate.weight, RLstate.stage, RLstate.available_action, RLstate.items_taken)

        # Available actions
        avail = env.get_valid_actions(RLstate)
        available = avail.astype(bool)

        # If no available action, it is the base case - calculate the true value
        if np.any(available) == False:
            if model == 'cappart':
                cost = moment_factors[0]*sum([means[j]*state[x[j]] for j in range(n)])-\
                        moment_factors[1]*(sum([deviations[j]*state[x[j]] for j in range(n)])**(1/2))+\
                        moment_factors[2]*(sum([skewnesses[j]*state[x[j]] for j in range(n)])**(1/3))-\
                        moment_factors[3]*(sum([kurtosis[j]*state[x[j]] for j in range(n)])**(1/4))
            elif model == 'narita':
                cost = 0.0
            v = cost * scaling_factor
            h = -v
            #print("h: ", h, state[stage])
            #print("")
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

        h = -v

        #print("")
        return h

    return RL_heuristic

def create_zero_g_function(total_weight, stage, is_pack, x, packed, 
                         means, deviations, skewnesses, kurtosis, moment_factors,
                         model_name='cappart'):

    def g_function(parent_g, state):
        if model_name == 'cappart':
            g = parent_g

        elif model_name == 'narita':
            g = moment_factors[0]*(sum([means[j] for j in state[packed]])+is_pack*means[state[stage]])-\
                    moment_factors[1]*((sum([deviations[j] for j in state[packed]])+is_pack*deviations[state[stage]])**(1/2))+\
                    moment_factors[2]*((sum([skewnesses[j] for j in state[packed]])+is_pack*skewnesses[state[stage]])**(1/3))-\
                    moment_factors[3]*((sum([kurtosis[j] for j in state[packed]])+is_pack*kurtosis[state[stage]])**(1/4))
        else:
            raise NotImplementedError

        return -g

    return g_function


def create_zero_heuristic_function():
    def zero_heuristic(state):
        return 0

    return zero_heuristic

def create_dual_heuristic_function(
        n, weights, capacity, 
        m, d, s, k, lmd,
        total_weight, stage,
        dual_bound=2
        ):

    check = [(lmd[0]*m[j] + lmd[2]*s[j]**(1/3)) / (weights[j] + 1e-6) for j in range(n)] + [0]
    print("check", check)

    max_efficiency = [max((lmd[0]*m[j] + lmd[2]*s[j]**(1/3)) / (weights[j] + 1e-6) for j in range(r, n)) for r in range(n)] + [0] # [0] is for the base case
    print("max_efficiency", max_efficiency)

    #NOTE: This is used only for debugging; for actual experiments, use regular DIDP features
    def dual_heuristic(state):
        if dual_bound == 0:
            return 0

        cur_weight = state[total_weight]
        cur_stage = state[stage]

        dual_bound_1 = 0
        dual_bound_2 = 0
        
        if dual_bound >= 1:
            remaining_m = lmd[0]*sum([m[j] for j in range(cur_stage, n)])
            remaining_s = lmd[2]*(sum([s[j] for j in range(cur_stage, n)])**(1/3))
            dual_bound_1 = remaining_m + remaining_s

        if dual_bound >= 2:
            dual_bound_2 = max_efficiency[cur_stage] * (capacity - cur_weight)


        h = -(min(dual_bound_1, dual_bound_2))

        #print("dual 1={}, dual 2={}, h={}, max_eff={}, rem_cap={}".format(
        #    dual_bound_1, dual_bound_2, h, max_efficiency[cur_stage], capacity - cur_weight))

        return h

    return dual_heuristic

def create_greedy_heuristic_function(n, weights, capacity, 
                                     m, d, s, k, lmd,
                                     total_weight, stage, 
                                     heuristic_type=1, debug=True):
    # heuristic_type (int): 1: use only means and skewness (same as dual bound), 2: use deviations too, 3: use kurtosis too, 4: use all

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
        # Portfolio: selecting the item with the highest "mean - deviation + skewness - kurtosis" divided by its weight

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

        return -h # maximization

    return greedy_heuristic

def create_solver(
    model, 
    total_weight, 
    stage, 
    x, 
    packed, 
    means, 
    deviations, 
    skewnesses, 
    kurtosis, 
    moment_factors, 
    n_item,
    capacity,
    weights,
    model_name,
    solver_name,
    scaling_factor=0.01,
    policy_name=None,
    initial_beam_size=1,
    time_limit=None,
    expansion_limit=None,
    discrete_coeff=0,
    greedy_heuristic_type=4,

    # RL heuristic
    heuristic='dual',
    heuristic_model_folder=None,
    policy_model_folder=None,
    softmax_temperature=1.0,
    input=None,
    train_n_item=None,
    train_lb=None,
    train_ub=None,
    train_capacity_ratio=None,
    train_moment_factors_0=None,
    train_moment_factors_1=None,
    train_moment_factors_2=None,
    train_moment_factors_3=None,
    seed=None,
    no_policy_accumulation=True,
    nouse_eta=1,
    nouse_g=1
):
    if heuristic == 'dqn':
        assert heuristic_model_folder is not None
        assert input is not None
        assert train_n_item is not None
        assert train_lb is not None
        assert train_ub is not None
        assert train_capacity_ratio is not None
        assert train_moment_factors_0 is not None
        assert train_moment_factors_1 is not None
        assert train_moment_factors_2 is not None
        assert train_moment_factors_3 is not None
        assert seed is not None

    if heuristic != 'dual' and policy_name == 'ppo':
        raise NotImplementedError
    elif heuristic != 'dual':
        if heuristic in ['dqn', 'ppoval']:
            g_evaluators = {}
            g_evaluators["invest"] = create_RL_g_function(
                    scaling_factor, total_weight, stage, 1, x, packed, 
                    means, deviations, skewnesses, kurtosis, moment_factors,
                    discrete_coeff=discrete_coeff, model_name=model_name
                )
            g_evaluators["ignore"] = create_RL_g_function(
                    scaling_factor, total_weight, stage, 0, x, packed, 
                    means, deviations, skewnesses, kurtosis, moment_factors,
                    discrete_coeff=discrete_coeff, model_name=model_name
                )

            h_evaluator = create_RL_heuristic_function(
                model_name, total_weight, stage, x, packed, means, deviations, skewnesses, kurtosis, moment_factors,
                n_item, capacity, weights, 
                heuristic_model_folder, input, train_n_item, train_capacity_ratio,
                train_moment_factors_0, train_moment_factors_1, 
                train_moment_factors_2, train_moment_factors_3,
                discrete_coeff, scaling_factor, seed, heuristic
            )
        elif heuristic == 'zero':
            g_evaluators = {}
            g_evaluators["invest"] = create_zero_g_function(
                    total_weight, stage, 1, x, packed, 
                    means, deviations, skewnesses, kurtosis, moment_factors,
                    model_name=model_name
                )
            g_evaluators["ignore"] = create_zero_g_function(
                    total_weight, stage, 0, x, packed, 
                    means, deviations, skewnesses, kurtosis, moment_factors,
                    model_name=model_name
                )

            h_evaluator = create_zero_heuristic_function()

        elif heuristic == 'greedy':
            g_evaluators = {}
            g_evaluators["invest"] = create_zero_g_function(
                    total_weight, stage, 1, x, packed, 
                    means, deviations, skewnesses, kurtosis, moment_factors,
                    model_name=model_name
                )
            g_evaluators["ignore"] = create_zero_g_function(
                    total_weight, stage, 0, x, packed, 
                    means, deviations, skewnesses, kurtosis, moment_factors,
                    model_name=model_name
                )

            h_evaluator = create_greedy_heuristic_function(
                n_item, weights, capacity, 
                means, deviations, skewnesses, kurtosis, moment_factors,
                total_weight, stage, 
                heuristic_type=greedy_heuristic_type, debug=True
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
        else:
            raise NotImplementedError
    elif policy_name == 'ppo':
        policy = create_RL_policy(model_name, total_weight, stage, x, packed, 
                                    n_item, capacity, weights, 
                                    policy_model_folder, input, train_n_item, train_capacity_ratio,
                                    train_moment_factors_0, train_moment_factors_1, 
                                    train_moment_factors_2, train_moment_factors_3,
                                    discrete_coeff, softmax_temperature, seed, 
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
                expansion_limit=expansion_limit
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
                expansion_limit=expansion_limit
            )
        if solver_name == "ACPS":
            solver = dp.UserPriorityACPS(
                model,
                policy=policy,
                policy_f_evaluator=policy_f_evaluator_noeta_nopi,
                time_limit=time_limit,
                quiet=False,
                no_policy_accumulation=no_policy_accumulation,
                expansion_limit=expansion_limit
            )
        if solver_name == "APPS":
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
        # Debugging
        """
        g_evaluators = {}
        g_evaluators["invest"] = create_zero_g_function(
                total_weight, stage, 1, x, packed, 
                means, deviations, skewnesses, kurtosis, moment_factors,
                model_name=model_name
            )
        g_evaluators["ignore"] = create_zero_g_function(
                total_weight, stage, 0, x, packed, 
                means, deviations, skewnesses, kurtosis, moment_factors,
                model_name=model_name
            )

        h_evaluator = create_dual_heuristic_function(
            n_item, weights, capacity, 
            means, deviations, skewnesses, kurtosis, moment_factors,
            total_weight, stage, 
            dual_bound=2)

        solver = dp.UserPriorityCABS(
            model,
            h_evaluator=h_evaluator,
            g_evaluators=g_evaluators,
            f_operator=dp.FOperator.Plus,  # g + h
            initial_beam_size=initial_beam_size,
            time_limit=time_limit,
            quiet=False,
            no_policy_accumulation=no_policy_accumulation,
            expansion_limit=expansion_limit
        )
        """

        solver = dp.CABS(
            model,
            initial_beam_size=initial_beam_size,
            time_limit=time_limit,
            quiet=False,
            expansion_limit=expansion_limit
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
    else:
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
    parser.add_argument("--policy-name", default="", type=str)
    parser.add_argument("--time-out", default=3600, type=int) # -1 means no time limit
    parser.add_argument("--expansion-limit", default=-1, type=int) # -1 means no node expansion limit
    parser.add_argument("--solver-name", default="CABS", type=str)
    parser.add_argument("--threads", default=1, type=int)
    parser.add_argument("--initial-beam-size", default=1, type=int)
    parser.add_argument("--parallel-type", default=0, type=int)
    parser.add_argument('--heuristic', type=str, default='dual') # dual, dqn, ppoval, zero, greedy
    parser.add_argument('--greedy-heuristic-type', type=int, default=4) # 1, 2, 3, 4
    parser.add_argument('--softmax-temperature', type=float, default=1.0)
    parser.add_argument('--no-policy-accumulation', default=1, type=int)

    # 0: no dual bound, 1: sum of remaining items, 2: 1 and max efficiency of remaining items
    parser.add_argument("--dual-bound", default=0, type=int)

    # Test instance parameters
    parser.add_argument("--model", default="cappart", type=str)
    parser.add_argument("--n-item", default=20, type=int)
    parser.add_argument("--lb", default=0, type=int)
    parser.add_argument("--ub", default=100, type=int)
    parser.add_argument("--capacity-ratio", default=0.5, type=float)
    parser.add_argument("--moment-factors-0", default=1, type=int)
    parser.add_argument("--moment-factors-1", default=5, type=int)
    parser.add_argument("--moment-factors-2", default=5, type=int)
    parser.add_argument("--moment-factors-3", default=5, type=int)
    parser.add_argument("--discrete-coeff", default=0, type=int)

    parser.add_argument("--num-instance", default=20, type=int)
    parser.add_argument('--seed', type=int, default=0)

    # RL model parameters
    parser.add_argument('--train-n-item', type=int, default=20)
    parser.add_argument('--train-lb', type=int, default=0)
    parser.add_argument('--train-ub', type=int, default=100)
    parser.add_argument('--train-capacity-ratio', type=float, default=0.5)
    parser.add_argument('--train-moment-factors-0', type=int, default=1)
    parser.add_argument('--train-moment-factors-1', type=int, default=5)
    parser.add_argument('--train-moment-factors-2', type=int, default=5)
    parser.add_argument('--train-moment-factors-3', type=int, default=5)
    parser.add_argument('--train-seed', type=int, default=0)

    parser.add_argument('--file', type=str, default="0.txt")
    parser.add_argument('--save', type=int, default=1) # save the solution to a file

    parser.add_argument('--nouse-eta', type=int, default=0) # 1 if dual is not used in policy f-evaluator
    parser.add_argument('--nouse-g', type=int, default=0) # 1 if g is not used in policy f-evaluator

    args = parser.parse_args()
    if args.time_out == -1:
        args.time_out = None
    if args.expansion_limit == -1:
        args.expansion_limit = None

    if args.heuristic not in ['dual', 'dqn', 'ppoval', 'zero', 'greedy']:
        raise NotImplementedError
    if args.heuristic == 'greedy':
        heuristic_name = args.heuristic + str(args.greedy_heuristic_type)
    else:
        heuristic_name = args.heuristic

    print("Instance: ", args.file)

    dataset_path = "n%d/lb%d-ub%d-cr%s-lmd%d%d%d%d-ni%d-s%d" % \
                        (args.n_item, args.lb, args.ub,                   
                         str(args.capacity_ratio).replace('.', ''),
                         args.moment_factors_0, args.moment_factors_1, 
                         args.moment_factors_2, args.moment_factors_3,
                         args.num_instance, args.seed)

    load_folder = os.path.join("./instances", "portfolio", dataset_path)
    history_folder = os.path.join(
                        "./history", 
                        "exlim%d" % args.expansion_limit if args.expansion_limit is not None else "no-exlim",
                        args.model, 
                        args.solver_name, 
                        "h-%s-%s" % (heuristic_name, args.policy_name), 
                        "portfolio",
                        "db%d%s%s" % (args.dual_bound, 
                                    "-eta0" if args.nouse_eta == 1 else "",
                                    "-g0" if args.nouse_g == 1 else ""),
                        dataset_path
                        )    

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

    n, capacity, weights, m, d, s, k, lmd = read_portfolio.read(os.path.join(load_folder, args.file))

    if args.model == "cappart":
        model, total_weight, i, x, means, deviations, skewnesses, kurtosis, moment_factors = \
            create_model_cappart(
                n, capacity, weights, m, d, s, k, lmd, 
                discrete_coeff=args.discrete_coeff
        )
        packed = None
    if args.model == "narita":
        model, total_weight, i, packed, means, deviations, skewnesses, kurtosis, moment_factors = \
            create_model_narita(
                n, capacity, weights, m, d, s, k, lmd, 
                discrete_coeff=args.discrete_coeff,
                dual_bound=args.dual_bound
        )
        x = None

    heuristic_model_folder = "./cp_rl_solver/selected-models/dqn/portfolio/%s/n-item-%d/capacity-ratio-%s/moment-factors-%d-%d-%d-%d" % \
                    (args.model, args.train_n_item,
                        str(args.train_capacity_ratio),
                        args.train_moment_factors_0, args.train_moment_factors_1, 
                        args.train_moment_factors_2, args.train_moment_factors_3)
    policy_model_folder = "./cp_rl_solver/selected-models/ppo/portfolio/%s/n-item-%d/capacity-ratio-%s/moment-factors-%d-%d-%d-%d" % \
                    (args.model, args.train_n_item, 
                        str(args.train_capacity_ratio), 
                        args.train_moment_factors_0, args.train_moment_factors_1, 
                        args.train_moment_factors_2, args.train_moment_factors_3)
    if args.heuristic == 'ppoval':
        heuristic_model_folder = policy_model_folder
    
    no_policy_accumulation = args.no_policy_accumulation == 1

    solver = create_solver(
        model, total_weight, i, x, packed, m, d, s, k, lmd,  
        args.n_item,
        capacity,
        weights,
        args.model,
        args.solver_name,
        scaling_factor=0.001,
        policy_name=args.policy_name,
        initial_beam_size=args.initial_beam_size,
        time_limit=args.time_out,
        expansion_limit=args.expansion_limit,
        greedy_heuristic_type=args.greedy_heuristic_type,
        # RL heuristic
        heuristic=args.heuristic,
        heuristic_model_folder=heuristic_model_folder,
        policy_model_folder=policy_model_folder,
        softmax_temperature=args.softmax_temperature,
        input=os.path.join(load_folder, args.file),
        train_n_item=args.train_n_item,
        train_lb=args.train_lb,
        train_ub=args.train_ub,
        train_capacity_ratio=args.train_capacity_ratio,
        train_moment_factors_0=args.train_moment_factors_0,
        train_moment_factors_1=args.train_moment_factors_1,
        train_moment_factors_2=args.train_moment_factors_2,
        train_moment_factors_3=args.train_moment_factors_3,
        seed=args.train_seed,
        no_policy_accumulation=no_policy_accumulation,
        nouse_eta=args.nouse_eta,
        nouse_g=args.nouse_g
    )

    investments, cost, summary = solve(
        solver,
        os.path.join(history_folder, "{}.csv".format(args.file.split('.')[0])),
        start_clock_time=start,
        save=args.save
    )

    solution = [1 if i in investments else 0 for i in range(n)]
    if cost is not None and read_portfolio.validate(args.n_item, capacity, weights, solution):
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
    
    #log_file.close()
