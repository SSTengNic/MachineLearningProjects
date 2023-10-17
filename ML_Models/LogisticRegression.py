import pandas as pd
import numpy as np
import matplotlib as plt
# Note that this will only work if y is either -1 or 1, and not 1 and 0.


def sigmoid(z):
    return 1/(1 + np.exp(-z))


# sgd refers to stochastic gradient descent!
def sgd_LG(features, labels, learning_rate=0.1, iter=5000):
    features_rows, features_columns = features.shape
    weight = np.zero(features_columns)
    offset = 0

    for _ in range(iter):
        # implement SGD
        # select t ∈ {1,...,n} at random
        rand_row = np.random.randint(features_rows)
        x_i, y_i = features[rand_row], labels[rand_row]

        s = y_i*(np.dot(weight, x_i + offset))
        # probability of Y, given X.
        gradient = (-y_i*x_i)/(1+np.exp(y_i*s))
        gradient_offset = -y_i / (1 + np.exp(y_i * s))

        weight = weight - (learning_rate * gradient)
        offset = offset - learning_rate * gradient_offset

    return weight, offset


def log_pred(weight, offset, test_features):
    # Calculate the predicted probabilities
    predictions = []
    for i in range(len(test_features)):
        s = np.dot(weight, test_features[i]) + offset
        prob = sigmoid(s)
        predictions.append(prob)

    # Convert probabilities to class labels (0 or 1)
    labels = [1 if p >= 0.5 else 0 for p in predictions]

    return labels
