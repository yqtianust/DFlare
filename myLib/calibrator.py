from sklearn import svm, tree
from sklearn.neural_network import MLPClassifier
import joblib


class Calibrator:
    def __init__(self, arch, load_path=None, random=0):
        self.model = None
        if load_path is None:
            if arch == "MLP":
                self.model = MLPClassifier(random_state=random, hidden_layer_sizes=50, max_iter=5000)
            elif arch == "SVM":
                self.model = svm.SVC(gamma=0.001)
            elif arch == "DT":
                self.model = tree.DecisionTreeClassifier()
        else:
            self.load_model(load_path)

    def save_model(self, save_path):
        joblib.dump(self.model, save_path)

    def load_model(self, load_path):
        self.model = joblib.load(load_path)
