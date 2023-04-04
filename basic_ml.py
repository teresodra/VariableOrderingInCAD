"""NOT IN USE"""


"""Contains a function to do some basic machine learning."""
import numpy as np
from tensorflow import keras
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
    elif model == 'my_mlp':
        return use_tf(x_train, x_test, y_train, y_test)
    else:
        raise Exception(f"No ML model called {model}")

    clf.fit(x_train, y_train)
    return accuracy_score(clf.predict(x_test), y_test)


def use_tf(x_train, x_test, y_train, y_test, batch_size=50, epochs=100):
    """Train a fully connected neural network and return its accuracy."""
    # design model
    model = keras.models.Sequential([
        keras.layers.Dense(80, activation='softmax',
                           input_shape=(x_train.shape[1],)),
        keras.layers.Dense(40, activation='softmax'),
        keras.layers.Dense(6, activation='softmax'),
    ])
    # choose hyperparameters
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optim = keras.optimizers.Adam(learning_rate=0.001)
    metrics = ["accuracy"]
    model.compile(loss=loss, optimizer=optim, metrics=metrics)

    # convert the type to float so that the numpy arrays
    # can be transformed into tensors
    x_train = np.asarray(x_train).astype('float32')
    x_test = np.asarray(x_test).astype('float32')

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    return model.evaluate(x_test, y_test, batch_size=batch_size)[1]
