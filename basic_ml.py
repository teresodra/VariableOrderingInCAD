"""Contains a function to do some basic machine learning."""
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def basic_ml(x_train, x_test, y_train, y_test, model, random_state=0):
    """
    Train the desired model and test its accuracy.

    The models are set to the hyperparameters that Dorian found in
    its paper regarding hyperparameter tuning.
    """
    if model == 'SVC':
        clf = svm.SVC(kernel='rbf',
                      C=1,
                      tol=0.0316,
                      random_state=random_state)
    elif model == 'DT':
        clf = DecisionTreeClassifier(
            splitter='best',
            criterion='gini',
            max_depth=1,
            random_state=random_state)
    elif model == 'KNN':
        clf = KNeighborsClassifier(
            n_neighbors=18,
            weights='distance',
            algorithm='ball_tree')
    elif model == 'RF':
        clf = RandomForestClassifier(n_estimators=10)
    elif model == 'MPL':
        clf = MLPClassifier(
            warm_start=False,
            hidden_layer_sizes=(1,),
            learning_rate='constant',
            solver='lbfgs',
            alpha=2.6366508987303556e-05,
            activation='logistic',
            max_iter=100000,
            random_state=random_state)
    else:
        raise Exception(f"No ML model called {model}")

    clf.fit(x_train, y_train)
    return accuracy_score(clf.predict(x_test), y_test)
