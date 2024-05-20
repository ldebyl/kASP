import random
import os
import numpy as np
import scipy.stats as stats
import math

def clip(x, min=None, max=None):
    if min and x < min:
        return min
    if max and x > max:
        return max
    return x

class RandomIterator:
    """
    Abstract Base Class for an iterator that generates
    a stream of random numbers, optionally bounded
    between a minimum and maximum. The random()
    function should be implemented for inheriting classes.
    """
    def __init__(self, min=None, max=None):
        self.min = min
        self.max = max

    def __iter__(self):
        return self
    
    def __next__(self):
        return clip(self.random(), self.min, self.max)
    
    def __call__(self):
        return self.random()

    def __int__(self):
        return int(self.random())
    
    def __float__(self):
        return float(self.random())
    
    def random(self):
        raise NotImplementedError()

class Uniform(RandomIterator):
    def __init__(self, min=0.0, max=1.0):
        self.min = min
        self.max = max

    def random(self):
        return np.random.uniform(self.min, self.max)

class UniformInteger(RandomIterator):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def random(self):
        return np.random.random_integers(self.min, self.max)

class Constant(RandomIterator):
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value

class RandomChoice(RandomIterator):
    def __init__(self, choices):
        self.choices = choices

    def random(self):
        return random.choice(self.choices)

class Normal(RandomIterator):
    """
    A RandomIterator that returns normally distributed
    floating point numbers with the specified mean
    and standard deviation.
    """
    def __init__(self, mean, std, **kwargs):
        super(self).__init__(**kwargs)
        self.mean = mean
        self.std = std

    def random(self):
        return np.random.normal(self.mean, self.std)

class Poisson(RandomIterator):
    def __init__(self, lam, **kwargs):
        super(self).__init__(**kwargs)
        self.lam = lam

    def random(self):
        return np.random.poisson(self.lam)

class TruncatedExponential(RandomIterator):
    def __init__(self, mu, a=0, b=1e10):
        self.mu = mu
        self.a = a
        self.b = b

    def random(self):
        "Samples from a truncated exponential distribution with bounds on [a, b) as integer"
        x = stats.truncexpon(b=(self.b-self.a)/self.mu, loc=self.a, scale=self.mu-self.a).rvs(1)[0]
        return math.floor(x)

class TruncatedNormal(RandomIterator):
    def __init__(self, mean, std, a=0, b=1e10):
        self.mean = mean
        self.std = std
        self.a = a
        self.b = b

    def random(self):
        "Samples from a truncated normal distribution with bounds on [a, b)"
        X = stats.truncnorm(
            (self.a - self.mean) / self.std, (self.b - self.mean) / self.std, loc=self.mean, scale=self.std)
        return X.rvs(size=1)[0]


class TruncatedGaussianMixture(RandomIterator):
    def __init__(self, means, stds, weights, a=0, b=1e10):
        self.means = means
        self.stds = stds
        self.weights = weights
        self.a = a
        self.b = b
        assert len(means) == len(stds) == len(weights), "The number of means, standard deviations, and weights must be equal"

    def random(self):
        "Samples from a truncated normal distribution with bounds on [a, b)"
        idx = random.choices(range(len(self.means)), weights=self.weights)[0]
        mu = self.means[idx]
        sigma = self.stds[idx]

        X = stats.truncnorm(
            (self.a - mu) / sigma, (self.b - mu) / sigma, loc=mu, scale=sigma)
        return X.rvs(size=1)[0]