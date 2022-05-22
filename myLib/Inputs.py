import pickle
import numpy as np

dataset_length = {"mnist": {"train": 50000, "test": 10000},
                  "cifar": {"train": 50000, "test": 10000},
                  "imagenet": 50000}


class Inputs:
    def __init__(self, dataset, train=False):
        if dataset not in dataset_length.keys():
            raise NotImplementedError
        self.dataset = dataset
        self.train = train

    def __getitem__(self, item):
        filename = "./seed_input/{}/{}/{:06}.npz".format(self.dataset, "train" if self.train else "test", item)
        seed_file = np.load(filename, allow_pickle=True)
        return seed_file

    # def __len__(self):
    #     return len(self)

    @property
    def len(self):
        return dataset_length[self.dataset]["train" if self.train else "test"]


class SubsetInputs:
    def __init__(self, dataset, source, arch, cps_type, random_seed):
        with open("./seed_inputs.p", "rb") as f:
            pickle_f = pickle.load(f)
            if dataset == "cifar":
                dataset = "cifar10"
            if cps_type == "kd":
                if arch == "vgg16":
                    cps_type += "-vgg11"
                else:
                    cps_type += "-simplenet"

            self.dataset = pickle_f[dataset][source][arch][cps_type][
                "output/random_seed_{}+50ps+512b".format(random_seed)]
            self.num_data = len(self.dataset["Y"])

    def __getitem__(self, item):
        img = self.dataset["X"][item, ...]
        label = self.dataset["Y"][item]
        seed_file = {"img": img, "label": label}
        return seed_file

    # def __len__(self):
    #     return len(self)

    @property
    def len(self):
        return self.num_data
