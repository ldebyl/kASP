import math

def find_closest_nCr(desired_value, max_n, min_r=2):
    desired_value = int(desired_value)
    max_n = int(max_n)
    min_delta = None
    optimal_parameters = None

    for n in range(max_n, min_r * 2 - 1, -1):
        for r in range(n // 2, min_r - 1, -1):
            # if the lower bound for nCr is already greater than desired_value, continue
            # oncrete Mathematics: A Foundation for Computer Science" by Ronald L. Graham, Donald E. Knuth, and Oren Patashnik (ISBN: 978-0201558029). In Chapter 5, Section 5.1 "Binomial Coefficients," the authors present the following inequality:

            if (n / r) ** r > desired_value:
                continue

            nCr = math.comb(n, r)
            delta = desired_value - nCr
            if delta < 0:
                # nCr > desired_value: continue decrementing r
                continue
            elif (min_delta is None or delta < min_delta):
                min_delta = delta
                optimal_parameters = (n, r)
            # Always break here - if we have reached this point, we have found the optimal r for this n
            # Continuing to decrease r will move further away from desired_value
            break
    assert optimal_parameters, "Could not find valid nCr values for desired complexity"
    assert math.comb(*optimal_parameters) <= desired_value
    return optimal_parameters

def find_nCr_from_complexity(complexity, nodes, max_n, min_r=2):
    """
    Given a complexity value, determine the optimal nCr values that will
    produce a value closest to the desired complexity.
    """
    N = complexity // nodes ** 2
    return find_closest_nCr(N, max_n, min_r)