"""IS THIS BEING USED?"""
import numpy as np
from sklearn.preprocessing import normalize


def convert_to_numpy_floats(features):
    return np.array([np.array([np.float64(feature)
                    for feature in feature_list])
                    for feature_list in features])


def normalize_features(features):
    """
    Normalize each column of features.

    The new media is 0 and the standard deviation is 1 in each column.
    """
    normal_features = []
    for feature in zip(*features):
        mean = np.mean(feature)
        std = np.std(feature)
        if std != 0:
            normal_features.append((feature - mean) / std)
        else:
            normal_features.append(feature - mean)
    return normal_features


def normalize_features2(features):
    """
    Normalize each column of features.

    The new media is 0 and the standard deviation is 1 in each column.
    """
    return normalize(features, axis=0)




# v = convert_to_numpy_floats([[2,1,4,1,41],[3,1,142,12,1],[21,12,34,123,2]])
# print(v[0,1])

# print(normalize_features2(v)==normalize_features(v))

# print(normalize_features2(v))
# print(normalize_features(v))