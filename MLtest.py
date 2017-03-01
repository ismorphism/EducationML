import MLalgorithms as ml
import numpy as np


X = np.matrix([1.1, 2.2, 3.3, 1, 7])
y = np.matrix(3.5)
clf = ml.LinearClassifier()
# clf.show()
clf.train(X, y)
# clf.show()
clf.predict([11, 2.2, 3.3, 10, 7])