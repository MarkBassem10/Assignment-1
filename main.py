import numpy as np


def randomization(n):

    z = np.random.random(size=n).reshape(n, 1)

    return z


def operations(h, w):

    X = np.random.random(size=(h, w))
    Y = np.random.random(size=(h, w))
    sum = X + Y

    return X, Y, sum


def norm(X, Y):

    sum = X + Y

    return np.linalg.norm(sum)


def neural_network(inputs, weights):

    i = np.tanh(weights.T.dot(inputs))

    return i


def scalar_function(a, b):

    if a<=b:
       return (np.dot(a,b))
    else:
       return(a/b)


def vector_function(a, b):

    vecfunc = np.vectorize(scalar_function(a,b))
    return vecfunc