import numpy as np
from numpy import exp as exp
import matplotlib.pyplot as plt



def sigm(X, k):
    y = np.asarray([1/(1 + exp(-k*float(i)))for i in X])
    return y


class LinearClassifier:

    def __init__(self):
        self.weigths = np.zeros((5,1))
        self.error = []

    def train(self, X, y):
        self.weigths = np.zeros(np.shape(X[0, :]))
        y_t = np.dot(X, np.transpose(self.weigths))
        self.error.append(float(0.5*np.dot(y - y_t, y - y_t)))
        alpha = 0.001
        i = 0
        max_iter = 100
        while i < max_iter-1:
            print(self.error[i])
            i += 1
            self.weigths += alpha*2*self.error[i - 1]*X
            y_t = np.dot(X, np.transpose(self.weigths))
            self.error.append(float(0.5 * np.dot(y - y_t, y - y_t)))
        print(y_t, ' ', self.error[max_iter - 1])
        x = np.arange(0, max_iter, 1)
        g = np.asarray(self.error)

        plt.setp(plt.plot(x, np.asarray(self.error)), color='b', linewidth=3.0)
        plt.show()

    def predict(self, X_test):
        y_test = np.matmul(X_test, np.transpose(self.weigths))
        print('Prediction is ', y_test)


