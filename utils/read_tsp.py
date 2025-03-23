import numpy as np

def read(filename):
    with open(filename) as f:
        values = f.read().split()

    position = 0
    n = int(values[position])
    position += 1
    nodes = list(range(n))
    edges = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                try:
                    edges[i, j] = int(values[position])
                except ValueError:
                    edges[i, j] = float(values[position])
            position += 1

    return n, nodes, edges

def read_xycoords(filename):
    # Same as read(), but returns x_coord and y_coord too

    with open(filename) as f:
        values = f.read().split()

    position = 0
    n = int(values[position])
    position += 1
    nodes = list(range(n))
    edges = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                try:
                    edges[i, j] = int(values[position])
                except ValueError:
                    edges[i, j] = float(values[position])
            position += 1

    x_coord = []
    for i in range(n):
        x_coord.append(float(values[position]))
        position += 1
    y_coord = []
    for i in range(n):
        y_coord.append(float(values[position]))
        position += 1

    return n, nodes, edges, x_coord, y_coord

def read_rl(filename):
    """
    Read an instance file that contains x_coord and y_coord and
    returns parameters readable by the DQN agent.
    
    """
    with open(filename) as f:
        values = f.read().split()

    position = 0
    n = int(values[position])
    position += 1    

    travel_time = []
    for i in range(n):
        tm = []
        for j in range(n):
            tm.append(int(values[position]))
            position += 1
        travel_time.append(tm)

    # TIme windows will be just ignored
    time_windows = np.zeros((n, 2))
    for i in range(n):
        time_windows[i, 0] = float(values[position])
        position += 1
        time_windows[i, 1] = float(values[position])
        position += 1
    
    x_coord = []
    for i in range(n):
        x_coord.append(float(values[position]))
        position += 1
    y_coord = []
    for i in range(n):
        y_coord.append(float(values[position]))
        position += 1

    return n, travel_time, x_coord, y_coord


def reduce_edges(nodes, edges, a, b):
    print("edges: {}".format(len(edges)))
    forward_dependent = []
    for (i, j) in edges.keys():
        if i == 0 or j == 0 or a[i] <= b[j]:
            forward_dependent.append((i, j))

    direct_forward_dependent = {}
    for (i, j) in forward_dependent:
        if (
            i == 0
            or j == 0
            or all(b[i] > a[k] or b[k] > a[j] for k in nodes if k != 0)
            or (a[i] == a[j] and b[i] == b[j])
        ):
            direct_forward_dependent[i, j] = edges[i, j]

    print("reduced edges: {}".format(len(direct_forward_dependent)))
    return direct_forward_dependent


def validate(n, edges, solution, cost, tolerance=1e-4, verbose=1):
    previous = solution[0]
    if previous != 0:
        if verbose: print(
            "The tour does not start from the depot {} but from {}".format(0, previous)
        )
        return False

    actual_cost = 0
    visited = set([0])

    for i in solution[1:-1]:
        if i < 0 or i > n - 1:
            if verbose: print("Customer {} does not exist".format(i))
            return False
        if i in visited:
            if verbose: print("Customer {} is already visited".format(i))
            return False
        visited.add(i)

        actual_cost += edges[previous, i]

        previous = i

    if solution[-1] != 0:
        if verbose: print(
            "The tour does not return to the depot {}, but to {}".format(
                0, solution[-1]
            )
        )
        return False

    actual_cost += edges[previous, 0]

    if len(visited) != n:
        if verbose: print(
            "The number of visited customers is {}, but should be {}".format(
                len(visited), n
            )
        )
        return False

    if abs(actual_cost - cost) > tolerance:
        if verbose: print(
            "The cost of the solution {} mismatches the actual cost {}".format(
                cost, actual_cost
            )
        )
        return False

    return True
