import ast
import torch
import numpy as np
import dgl

def string_to_list(input_string):
    """
    Convert a string representation of a list to a list object.
    input_string (str): string representation of a list, e.g., "[1, 2, 3]".
    
    """
    try:
        # Safely evaluate the string as a Python literal expression
        result_list = ast.literal_eval(input_string)
        if isinstance(result_list, list):
            return result_list
        else:
            raise ValueError("Input is not a valid list representation.")
    except (SyntaxError, ValueError) as e:
        print("Error:", e)
        return None


def compute_h(RLstate, agent, env, problem_name=None):
    # Available actions
    avail = env.get_valid_actions(RLstate)
    available = avail.astype(bool)

    # If no available action, it is the base case - calculate the true value
    if np.any(available) == False:
        return 0.0

    nn_input = env.make_nn_input(RLstate, 'cpu')

    if problem_name in ['tsp', 'tsptw']:
        nn_input = dgl.batch([nn_input])
        # Get RL prediction
        with torch.no_grad():
            res = agent.model(nn_input, graph_pooling=False)

        res = dgl.unbatch(res)
        out = [r.ndata["n_feat"].data.cpu().numpy().flatten() for r in res]
        out = out[0].reshape(-1)
    else:
        nn_input = nn_input.unsqueeze(0)
        # Get RL prediction
        with torch.no_grad():
            res = agent.model(nn_input)

        out = res.cpu().numpy().squeeze(0)

    # Mask unavailable actions
    v = np.max(out[available])

    h = -v

    return h

def compute_policy(RLstate, agent, env, 
                   non_zero_base_case=False, softmax_temperature=1.0):
    #print("state", state[unvisited], state[location], state[time_var])
    unvisited_set = RLstate.must_visit

    depot_prob = [np.float64("nan")] 
    if not non_zero_base_case and len(unvisited_set) == 0:
        depot_prob = [1.0] # depot is a forced transition, so must come at the end

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
            for j in range(1, len(action_probs))
        ] + depot_prob,
        dtype=np.float64,
    )

    # Policy must return log probabilities
    log_probabilities = np.log(probabilities)

    return log_probabilities

def compute_q(RLstate, agent, env):
    # Available actions
    avail = env.get_valid_actions(RLstate)
    available = avail.astype(bool)

    # If no available action, the value is 0.0 (no reward can be obtained)
    if np.any(available) == False:
        return [0.0 for _ in range(len(available))]

    nn_input = env.make_nn_input(RLstate, 'cpu')
    nn_input = nn_input.unsqueeze(0)

    # Get RL prediction
    with torch.no_grad():
        res = agent.model(nn_input)

    out = res.cpu().numpy().squeeze(0)

    # If the action is unavailable, the value is -inf
    out[~available] = -np.inf

    return out