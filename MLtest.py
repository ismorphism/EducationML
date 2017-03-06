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
print(data.keys())
print(label_names['label_names'])
dict = label_names['label_names']
X = data['data'][0, :]
y = np.zeros((10,))
y[data['labels'][0]] = 1

clf = ml.LinearClassifier(y, X)
plt.ion()

for i in range(10000):
    X = data['data'][i, :]
    y = np.zeros((10,))
    y[data['labels'][i]] = 1
    clf.train(X, y)
    # print(i)
    number = clf.output.index(1)
    print(dict[number])
    # img = Image.fromarray(X.reshape((32, 32, 3)), 'RGB')
    # img.save('my.png')
    # img.show()
    # plt.imshow(X.reshape((32, 32, 3)))
    # plt.show()
    # plt.pause(2)

print(clf.weigths)

total_err = 0
for i in range(100, 200):
    y = np.zeros((10,))
    y[data['labels'][i]] = 1
    X_test = data['data'][i, :]
    pred = clf.predict(X_test)
    try:
        pred_number = pred.index(1)
        print('True is ', dict[data['labels'][i]], 'Prediction is ', dict[pred_number])
    except ValueError:
        pass
    total_err += sum([abs(i) for i in (clf.predict(X_test) - y)])
print(total_err)