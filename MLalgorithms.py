import numpy as np
from numpy import exp as exp
import matplotlib.pyplot as plt
from scipy import special as sp
from decimal import *
# getcontext().prec = 25

def sigm(X, k=1000):
    y = [float((1/(1 + exp(-k*i)))) for i in X]
    return y


class LinearClassifier:

    def __init__(self, y, X):
        self.weigths = np.zeros((len(y), len(X)))
        self.error = []

    def train(self, X, y):
        y_t = np.dot(X, np.transpose(self.weigths))
        y_t = sigm(y_t)
        self.error.append((y - y_t))
        alpha = 0.1
        i = 0
        while sum([abs(j) for j in self.error[i]]) > 0:
            # print(self.error[i])
            i += 1
            for j in range(len(y)):
                self.weigths[j, :] += alpha*2*np.dot(sum([i[j] for i in self.error]), X)
            y_t = np.dot(X, np.transpose(self.weigths))
            y_t = sigm(y_t)
            self.error.append((y - y_t))
        # print(y_t, ' ', [sum(j) for j in self.error])
        self.output = y_t
        self.error = []
        x = np.arange(0, i + 1, 1)
        g = np.asarray(self.error)

        # plt.setp(plt.plot(x, np.asarray([sum(j) for j in self.error])), color='b', linewidth=3.0)
        # plt.show()

    def show(self):
        print(self.weigths)

    def predict(self, X_test):
        y_p = np.dot(X_test, np.transpose(self.weigths))
        y_p = sigm(y_p)
        # print('Prediction is ', y_p)
        return y_p

# print(sigm([10, 50, 5, 65, 99], 1))
# print(np.shape(np.matrix([1,2,3]))[1])