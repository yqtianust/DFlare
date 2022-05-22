import numpy as np
from myUtils.others import get_timestamp


class MutationP:
    def __init__(self, m, total=0, delta_bigger_than_zero=0, epsilon=1e-7):
        self.mutation_method = m
        if hasattr(self.mutation_method, "__name__"):
            self.name = self.mutation_method.__name__
        else:
            self.name = get_timestamp()
        self.total = total
        self.delta_bigger_than_zero = delta_bigger_than_zero
        self.epsilon = epsilon

    def mut(self, img):
        return np.clip(self.mutation_method(img), 0, 255)

    @property
    def score(self, epsilon=1e-7):
        # mylogger = Logger()
        rate = self.delta_bigger_than_zero / (self.total + epsilon)
        # mylogger.info("Name:{}, rate:{}".format(self.name, rate))
        return rate


class ProbabilityImgMutations:

    def __init__(self, ops, random_seed):
        self.p = 1 / len(ops)
        self.mutation_method = [MutationP(m) for m in ops]
        self.random = np.random.RandomState(random_seed)
        self.num_mutation_method = len(self.mutation_method)

    def add_mutation(self, m):
        if not isinstance(m, MutationP):
            m = MutationP(m)
        self.mutation_method.append(m)
        self.num_mutation_method = len(self.mutation_method)

    @property
    def mutators(self):
        mus = {}
        for mu in self.mutation_method:
            mus[mu.name] = mu
        return mus

    def choose_mutator(self, mu1=None):
        if mu1 is None:
            # which means it's the first mutation
            return self.mutation_method[np.random.randint(0, self.num_mutation_method)]
        else:
            self.sort_mutators()
            k1 = self.index(mu1)
            k2 = -1
            prob = 0
            while self.random.rand() >= prob:
                k2 = self.random.randint(0, self.num_mutation_method)
                prob = (1 - self.p) ** (k2 - k1)
            mu2 = self.mutation_method[k2]
            return mu2

    def sort_mutators(self):
        import random
        random.shuffle(self.mutation_method)
        self.mutation_method.sort(key=lambda mutator: mutator.score, reverse=True)

    def index(self, mutator_name):
        for i, mu in enumerate(self.mutation_method):
            if mu.name == mutator_name:
                return i
        return -1


class RandomImgMutations(ProbabilityImgMutations):
    def choose_mutator(self, mu1=None):
        idx = self.random.randint(low=0, high=self.num_mutation_method)
        return self.mutation_method[idx]
