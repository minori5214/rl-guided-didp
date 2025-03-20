def read(filename):
    with open(filename) as f:
        values = f.read().split()

    position = 0
    n = int(values[position])
    position += 1
    capacity = int(values[position])
    position += 1

    weight_list = []
    for i in range(n):
        weight_list.append(float(values[position]))
        position += 1

    profit_list = []
    for i in range(n):
        profit_list.append(float(values[position]))
        position += 1

    return n, capacity, weight_list, profit_list

def validate(n_item, capacity, weights, profits, solution, cost, tolerance=1e-4):
    if n_item != len(solution):
        return False

    total_weight = 0
    total_profit = 0
    packed = [i for i, x in enumerate(solution) if x == 1]

    for i in packed:
        total_weight += weights[i]
        total_profit += profits[i]

    if total_weight > capacity:
        print("Total weight ({}) exceeds knapsack capacity ({})".format(
            total_weight, capacity)
        )
        return False
    
    if abs(total_profit - cost) > tolerance:
        print("Total profit ({}) does not match cost ({})".format(
            total_profit, cost)
        )
        return False

    return True