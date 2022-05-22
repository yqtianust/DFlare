class StateFitnessValue:
    def __init__(self, add_to_corpus, diff_value):
        self.add_to_corpus = add_to_corpus
        self.diff_value = diff_value

    def better_than(self, other):
        if abs(abs(self.diff_value) - abs(other.diff_value)) > 1e-3:
            return True
        else:
            return self.add_to_corpus


    def __str__(self):
        return "Fitness value: {} {:.6f}".format(self.add_to_corpus, self.diff_value)


class DiffProbFitnessValue:
    def __init__(self, diff_value, dummy=0):
        self.diff_value = diff_value

    def better_than(self, other):
        return abs(self.diff_value) > abs(other.diff_value)

    def __str__(self):
        return "Fitness value: {:.6f}".format(self.diff_value)
