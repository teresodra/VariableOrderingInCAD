"""Contains the ml models that will be used in the project"""

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

ml_models = [
            #  'KNN',
            #  'DT',
            #  'SVC',
             'RF',
            #  'MLP'
             ]

ml_regressors = [
                #  'DTR',
                #  'SVR',
                 'RFR',
                #  'KNNR',
                #  'MLPR'
                 ]

sklearn_models = {
    'DT': DecisionTreeClassifier,
    'KNN': KNeighborsClassifier,
    'RF': RandomForestClassifier,
    'SVC': SVC,
    'MLP': MLPClassifier,
    'DTR': DecisionTreeRegressor,
    'KNNR': KNeighborsRegressor,
    'RFR': RandomForestRegressor,
    'SVR': SVR,
    'MLPR': MLPRegressor
}
