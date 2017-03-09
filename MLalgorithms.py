import numpy as np
from numpy import exp as exp
import matplotlib.pyplot as plt
from scipy import special as sp
from decimal import *
# getcontext().prec = 25

def sigm(X, k=1000):
    n = np.shape(X)[0]
    m = np.shape(X)[1]
    A = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            A[i][j] = 1/(1 + exp(-k*X[i, j]))
            # A[i][j] = ( 1 / (1 + exp(-k * X[i, j])) - 0.5)*2
    return np.matrix(A)

def err_sum_col(X):
    n = np.shape(X)[0]
    A = np.zeros((n, 1))
    for i in range(n):
        A[i] = sum(np.array(X[i,:])[0])
    return A

def cross_entropy(Y_pred, Y_true):
    n = np.shape(Y_pred)[0]
    m = np.shape(Y_pred)[1]
    A = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            # A[i][j] = -Y_true[i, j]*np.log(Y_pred[i, j]) - (1 - Y_true[i, j])*np.log(1 - Y_pred[i, j])
            A[i][j] = np.log(1 + exp(-1*Y_true[i, j]*Y_pred[i, j]))
    return np.matrix(A)


class LinearClassifier:

    def __init__(self, X, y):
        self.weigths = np.zeros((len(y), np.shape(X)[1]))

    def train(self, X, y):
        y_t = np.matmul(self.weigths, np.transpose(X))
        y_t = sigm(y_t)
        err = (y - y_t)
        # err = cross_entropy(y_t, y)
        alpha = 0.001
        i = 0
        while i < 250:
            i += 1
            k = np.matmul(np.matrix(err), np.matrix(X))
            self.weigths += alpha*k
            y_t = np.matmul(self.weigths, np.transpose(X))
            y_t = sigm(y_t)
            err = (y - y_t)
            # err = cross_entropy(y_t, y)
        print(y_t)
        self.outputs = y_t

    def show(self):
        print(self.weigths)

    def predict(self, X_test):
        y_p = np.matmul(self.weigths, np.transpose(X_test))
        y_p = sigm(y_p)
        return y_p


print(cross_entropy(np.matrix([1, 1, 1]), np.matrix([-1, 1, -1])))
