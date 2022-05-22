import os
import numpy as np
import time
import pickle


def get_label_and_prob(output):
    if type(output) is not np.ndarray:
        output = output.numpy()

    argmin = np.argsort(output, axis=1)
    label = argmin[:, -1]
    prob = output[np.arange(0, len(label)), label]
    return label[0], prob


class PredictResult:
    def __init__(self, prediction_vector):
        self.vec = prediction_vector
        self.label, self.prob = get_label_and_prob(self.vec)


class BaseResult:

    def __init__(self, seed_input: np.array, seed_label: int, idx: int,
                 org_result: PredictResult, cps_result: PredictResult,
                 output_dir: str):
        self.seed_input = seed_input
        self.seed_label = seed_label
        self.idx = idx
        self.seed_org_result = org_result
        self.seed_cps_result = cps_result
        self.output_dir = output_dir

        self.save_time = 0

    def save(self):
        self.save_time = time.time()

        filename = "{:06}.pickle".format(self.idx)

        with open(os.path.join(self.output_dir, filename), 'wb') as outfile:
            pickle.dump(self, outfile, pickle.HIGHEST_PROTOCOL)


class Result(BaseResult):

    def __init__(self, seed_input: np.array, seed_label: int, idx: int,
                 org_result: PredictResult, cps_result: PredictResult,
                 output_dir: str):
        super().__init__(seed_input, seed_label, idx, org_result, cps_result, output_dir)

        self.last_mutant = {"org": None, "cps": None}
        self.last_org_result = {"org": None, "cps": None}
        self.last_cps_result = {"org": None, "cps": None}

        self.mutant = {"org": None, "cps": None}
        self.org_result = {"org": None, "cps": None}
        self.cps_result = {"org": None, "cps": None}

        self.iteration = {"org": -1, "cps": -1}
        self.start_time = {"org": 0.0, "cps": 0}
        self.end_time = {"org": 0, "cps": 0}

    def set_start_time(self, targeted_model):
        self.start_time[targeted_model] = time.time()

    def update_last_mutant(self, mutant, org_result: PredictResult, cps_result: PredictResult, targeted_model: str):
        self.last_mutant[targeted_model] = np.copy(mutant)
        self.last_org_result[targeted_model] = org_result
        self.last_cps_result[targeted_model] = cps_result

    def update_results(self, mutant, org_result: PredictResult, cps_result: PredictResult, targeted_model: str,
                       iteration: int):
        self.mutant[targeted_model] = np.copy(mutant)
        self.org_result[targeted_model] = org_result
        self.cps_result[targeted_model] = cps_result

        self.iteration[targeted_model] = iteration
        self.end_time[targeted_model] = time.time()


class SingleAttackResult(BaseResult):

    def __init__(self, seed_input: np.array, seed_label: int, idx: int,
                 org_result: PredictResult, cps_result: PredictResult,
                 output_dir: str):
        super().__init__(seed_input, seed_label, idx, org_result, cps_result, output_dir)

        self.mutant = None
        self.org_result = None
        self.cps_result = None
        self.iteration = -1
        self.start_time = time.time()
        self.end_time = 0

    def update_results(self, mutant, org_result, cps_result, iteration):
        self.mutant = mutant
        self.org_result = org_result
        self.cps_result = cps_result
        self.iteration = iteration
        self.end_time = time.time()

    def update_reduced_results(self, mutant, org_result, cps_result, iteration, l2, linf):
        self.reduced_mutant = mutant
        self.reduced_org_result = org_result
        self.reduced_cps_result = cps_result
        self.reduced_iteration = iteration
        self.l2 = l2
        self.linf = linf
        self.reduced_end_time = time.time()
