import numpy as np
from scipy.stats import uniform
from sklearn import svm
import pickle

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def quick_hyperparams_check(x_train, y_train, x_test, y_test):
    cs = np.logspace(-2, 4, 7)
    gammas = np.logspace(-3, 2, 6)

    for c in cs:
        clf = SVC(C=c, kernel="linear")
        clf.fit(x_train, y_train)
        print(f"linear c={c}: {clf.score(x_test, y_test)}")

    for c in cs:
        for gamma in gammas:
            clf = SVC(C=c, gamma=gamma, kernel="rbf")
            clf.fit(x_train, y_train)
            print(f"rbf c={c}, gamma={gamma}: {clf.score(x_test, y_test)}")

    for c in cs:
        for gamma in gammas:
            if c * gamma > 100:
                continue
            clf = SVC(C=c, gamma=gamma, degree=2, kernel="poly")
            clf.fit(x_train, y_train)
            print(f"poly c={c}, gamma={gamma}, degree=2: {clf.score(x_test, y_test)}")

def get_data():
    x_train, y_train, x_test, y_test = [], [], [], []
    with open("../transform/training.txt", "r") as file:
        for line in file:
            data = [float(item) for item in line.split()]
            x_train.append(data[1:])
            y_train.append(data[0])
    with open("../transform/test.txt", "r") as file:
        for line in file:
            data = [float(item) for item in line.split()]
            x_test.append(data[1:])
            y_test.append(data[0])
    return x_train, y_train, x_test, y_test

def save(clf):
    with open("model.pkl", "wb") as file:
        pickle.dump(clf, file)

def train(x_train, y_train, x_test, y_test):
    param_grid = [
        {'C': np.linspace(100, 400, num=10), 'kernel': ['linear']},
        {'C': [100], 'gamma': np.logspace(-1.5, 0, num=100), 'kernel': ['rbf']},
        {'C': [10], 'degree': [2], 'gamma': np.linspace(5, 12, num=8), 'kernel': ['poly']},
    ]
    clf = GridSearchCV(svm.SVC(), param_grid, n_jobs=-1)
    clf.fit(x_train, y_train)
    # for kernel, c, gamma, score in zip(clf.cv_results_["param_kernel"], clf.cv_results_["param_C"],
    #                                    clf.cv_results_["param_gamma"], clf.cv_results_["mean_test_score"]):
        # print(f"{kernel} c={c}, gamma={gamma}: accuracy {score}")
        # pass
    # print(clf.coef_)
    print(clf.score(x_test, y_test))
    print(clf.best_params_)
    save(clf)

def train_no_time(x_train, y_train, x_test, y_test):
    c=100
    gamma=0.06353752638084484
    clf = SVC(C=c, gamma=gamma, degree=2, kernel="rbf")
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))
    save(clf)

def train_time_series(x_train, y_train, x_test, y_test):
    c=1.0
    gamma=1e-07
    clf = SVC(C=c, gamma=gamma, degree=2, kernel="poly")
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))
    save(clf)

def benchmark(x_train, y_train, x_test, y_test):
    classifiers = [
        SVC(),
        SVC(kernel="linear"),
        RandomizedSearchCV(SVC(), dict(C=uniform(0, 10)), n_jobs=-1, random_state=0),
        MLPClassifier(alpha=1, max_iter=1000, random_state=42),
        GradientBoostingClassifier(random_state=0)
    ]

    for clf in classifiers:
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(x_train, y_train)
        print(clf.score(x_test, y_test))

def main():
    x_train, y_train, x_test, y_test = get_data()
    #benchmark(x_train, y_train, x_test, y_test)
    # quick_hyperparams_check(x_train, y_train, x_test, y_test)
    # train_no_time(x_train, y_train, x_test, y_test)
    train_time_series(x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    main()