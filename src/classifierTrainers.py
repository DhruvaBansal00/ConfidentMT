from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
import numpy as np
from itertools import zip_longest
import matplotlib.pyplot as plt
import random
from sklearn.metrics import auc
import statistics
from tqdm.notebook import tqdm

random.seed(42)

class CustomEnsembleClassifier:
    def __init__(self, clfs):
        self.classifiers = clfs
    
    def predict(self, X):
        probabilities = None
        for clf in self.classifiers:
            if probabilities is None:
                probabilities = clf.predict_proba(X)
            else:
                probabilities += clf.predict_proba(X)
        return np.argmax(np.array(probabilities), axis=1)

def trainLogisticRegressionClassifier(X, Y, verbose=False):
    if verbose:
        print("Traning Logistic Regression Classifier")
    clf = LogisticRegression(random_state=42)
    clf.fit(X, Y)
    return clf

def trainMLPClassifier(X, Y, verbose=False):
    if verbose:
        print("Training MLP Classifier")
    clf = MLPClassifier(hidden_layer_sizes=(256, 1024, 2048, 1024, 256), random_state=42,
                        max_iter=200, learning_rate='adaptive', learning_rate_init=0.0001, activation='relu',
                        verbose=verbose)
    clf.fit(X, Y)
    return clf

def trainKNeighborsClassifier(X, Y, verbose=False):
    if verbose:
        print("Training KNeighbors Classifier")
    clf = KNeighborsClassifier(100)
    clf.fit(X, Y)
    return clf

def trainGaussianProcessClassifier(X, Y, verbose=False):
    if verbose:
        print("Training Gaussian Process Classifier")
    length_scale = [1 for i in range(len(X[0]))]
    clf = GaussianProcessClassifier(1.0 * RBF(length_scale), warm_start=True, random_state=42, n_jobs=-1)
    clf.fit(X, Y)
    return clf

def trainCustomEnsemble(X, Y, maxDepth=8, estimators=100, verbose=False, subsetTraining=False):
    if verbose:
        print("Training custom ensemble")
    rf = RandomForestClassifier(max_depth=maxDepth, random_state=42)
    grad = GradientBoostingClassifier(random_state=42)
    ada = AdaBoostClassifier(n_estimators=estimators, random_state=42)
    kn = KNeighborsClassifier(100)
    lg = LogisticRegression(random_state=42)

    classifiers = [rf, grad, ada, kn, lg]

    Xs = [X for i in range(len(classifiers))]
    Ys = [Y for i in range(len(classifiers))]

    if subsetTraining:
        Xs = []
        Ys = []
        data = list(zip(X, Y))
        random.shuffle(data)
        X_all, Y_all = zip(*data)

        start = 0
        end = int(len(X)/len(classifiers))

        for i in range(len(classifiers)):
            Xs.append(X_all[start:end])
            Ys.append(Y_all[start:end])
            start = end
            end = int((i+2) * len(X)/len(classifiers))

    for index in range(len(classifiers)):
        classifiers[index].fit(Xs[index], Ys[index])

    return CustomEnsembleClassifier(classifiers)
    

def trainEnsembleClassifier(X, Y, maxDepth=8, estimators=100, verbose=False):
    if verbose:
        print("Training an ensemble of Random Forest and Gradient Boosting Classifiers")

    estimators = [
     ('rf', RandomForestClassifier(max_depth=maxDepth, random_state=42)),
     ('grad', GradientBoostingClassifier(random_state=42))]
    clf = StackingClassifier(estimators=estimators, final_estimator=AdaBoostClassifier(n_estimators=50, random_state=42))
    clf.fit(X, Y)
    return clf


def trainRandomForestClassifier(X, Y, maxDepth=8, verbose=False):
    if verbose:
        print("Training Random Forest classifier")
    clf = RandomForestClassifier(max_depth=maxDepth, random_state=42)
    clf.fit(X, Y)
    return clf

def trainAdaBoostClassifier(X, Y, estimators=100, verbose=False):
    if verbose:
        print("Training AdaBoosted Decision Tree classifier")
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=estimators, random_state=42)
    clf.fit(X, Y)
    return clf

def trainGradientBoostingClassifier(X, Y, verbose=False):
    if verbose:
        print("Training Graident Boosted classifier")
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X, Y)
    return clf

def trainSVM(X, Y, verbose=False):
    if verbose:
        print("Training SVM classifier")
    clf = SVC(gamma='auto')
    clf.fit(X, Y)
    return clf