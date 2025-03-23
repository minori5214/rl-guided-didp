def read(filename):
    with open(filename) as f:
        values = f.read().split()

    position = 0
    n = int(values[position])
    position += 1
    capacity = int(values[position])
    position += 1

    weights = []
    for i in range(n):
        weights.append(int(values[position]))
        position += 1

    means = []
    for i in range(n):
        means.append(int(values[position]))
        position += 1

    deviations = []
    for i in range(n):
        deviations.append(int(values[position]))
        position += 1

    skewnesses = []
    for i in range(n):
        skewnesses.append(int(values[position]))
        position += 1

    kurtosis = []
    for i in range(n):
        kurtosis.append(int(values[position]))
        position += 1

    moment_factors = []
    for i in range(4):
        moment_factors.append(int(values[position]))
        position += 1

    return n, capacity, weights, means, deviations, skewnesses, kurtosis, moment_factors

def validate(n_item, capacity, weights, solution):
    if n_item != len(solution):
        print("n_item {} and solution length {} do not match.".format(n_item, len(solution)))
        return False

    total_weight = 0
    investments = [i for i, x in enumerate(solution) if x == 1]
    for i in investments:
        total_weight += weights[i]
    #print("Total weight:", total_weight)
    #print("Capacity:", capacity)
    return total_weight <= capacity

def get_profit(solution, n_item, means, deviations, skewnesses, kurtosis, moment_factors, capacity, weights):
    if validate(n_item, capacity, weights, solution) == False:
        return None
    total_profit = moment_factors[0]*sum([means[j]*solution[j] for j in range(n_item)])-\
                    moment_factors[1]*(sum([deviations[j]*solution[j] for j in range(n_item)])**(1/2))+\
                    moment_factors[2]*(sum([skewnesses[j]*solution[j] for j in range(n_item)])**(1/3))-\
                    moment_factors[3]*(sum([kurtosis[j]*solution[j] for j in range(n_item)])**(1/4))
    return total_profit