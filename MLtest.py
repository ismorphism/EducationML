import MLalgorithms as ml
import numpy as np
import pickle as cPickle
import matplotlib.pyplot as plt
from PIL import Image

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo, encoding='latin')
    fo.close()
    return dict

data = unpickle("cifar-10-batches-py/data_batch_1")
label_names = unpickle("cifar-10-batches-py/batches.meta")
dict = label_names['label_names']
X = data['data'][0, :]
y = np.zeros((10,))
y[data['labels'][0]] = 1

data_test = unpickle("cifar-10-batches-py/test_batch")

X_train = np.zeros((1, len(X)))
y_train = np.zeros(np.shape(y))

clf = ml.LinearClassifier(X_train, y_train)

for i in range(2000):
    X = np.matrix(data['data'][i, :])
    X = (X - np.mean(X))/np.max(X)
    X_train = np.r_[X_train, X]
    y = np.zeros((10,))
    y[data['labels'][i]] = 1
    y_train = np.c_[y_train, y]

H = X_train[1:, :]
h = y_train[:, 1:]


clf.train(H, h)


# clf.train(X_train, y_train)
    # print(i)
    # number = clf.output.index(1)
    # print(dict[clf.outputs.index(1)])
    # img = Image.fromarray(X.reshape((32, 32, 3)), 'RGB')
    # img.save('my.png')
    # img.show()
    # plt.imshow(X.reshape((32, 32, 3)))
    # plt.show()
    # plt.pause(2)


total_err = 0
for i in range(1, 31):
    X_test = np.matrix(data_test['data'][i, :])
    pred = clf.predict(X_test).flatten()
    try:
        print('True is ', dict[data_test['labels'][i]], 'Prediction is ', dict[pred.tolist()[0].index(1)])
        if dict[data_test['labels'][i]] != dict[pred.tolist()[0].index(1)]:
            total_err += 1
    except ValueError:
        total_err += 1
        pass

print(total_err)
