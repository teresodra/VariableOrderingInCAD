"""Contains the ml models that will be used in the project"""

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor


classifiers = {
    'DT-Classifier': DecisionTreeClassifier,
    'KNN-Classifier': KNeighborsClassifier,
    'RF-Classifier': RandomForestClassifier,
    'SVM-Classifier': SVC,
    'MLP-Classifier': MLPClassifier,
    # 'GB-Classifier': GradientBoostingClassifier
}

regressors = {
    'DT-Regressor': DecisionTreeRegressor,
    'KNN-Regressor': KNeighborsRegressor,
    'RF-Regressor': RandomForestRegressor,
    'SVM-Regressor': SVR,
    'MLP-Regressor': MLPRegressor,
    # 'GB-Regressor': GradientBoostingRegressor
}

all_models = {**classifiers, **regressors}

heuristics = ['T1', 'gmods', 'brown', 'random', 'virtual-best']
