"""Contains the grid of hyperparameters that each model will try"""

grid = dict()
grid['RF'] = {
    'n_estimators': [200, 300, 400, 500],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}
grid['KNN'] = {
    'n_neighbors': [1,3,5,7,12],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    #'leaf_size': range(1, 10, 3),
    #'p': range(1, 4, 1)
}
grid['MLP'] = {
    'hidden_layer_sizes': [(5,5), (15,15), (20,20), (10,10,10), (20,20,20)], #[(i,i) for i in range(50, 20, 5)],# +[(i,i, i) for i in range(50, 20, 5)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'learning_rate': ['constant','adaptive'],
    'alpha': [0.05, 0.005],
    'max_iter': [1000]
}
grid['DT'] = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [1,4,7,10,13,16,19]
}
grid['SVC'] = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'tol': [0.0316],
    'C': [5,100,200,300],
    'gamma': ['scale', 'auto']
}

grid['RFR'] = {
    'criterion': ['squared_error', 'friedman_mse'],
    "max_depth": [1,3,7],
    "min_samples_leaf": [1,5,10],
}
grid['KNNR'] = {
    'n_neighbors': [3, 5, 10],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}
grid['MLPR'] = {
    'hidden_layer_sizes': [(100,), (20, 20), (10, 10, 10)],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01]
}
grid['DTR'] = {
    "splitter":["best","random"],
    "max_depth" : [1,3,7,12],
    "min_samples_leaf":[1,5,10],
    # "min_weight_fraction_leaf":[0.1,0.5,0.9],
    # "max_features":["auto","log2","sqrt",None],
    # "max_leaf_nodes":[None,10,50,90]
}
grid['SVR'] = {
    'kernel': ('linear', 'rbf','poly'),
    'C':[1.5, 10],
    'gamma': [1e-7, 1e-4],
    'epsilon':[0.1,0.2,0.5,0.3]
}
grid['SGD'] = {
    'loss':["squared_error", "huber", "epsilon_insensitive"],
    'penalty':["l2", "l1", "elasticnet"]
}
