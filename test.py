import numpy as np
from Net import Net
from Net import _change_one_hot_label

network = Net(input_size=784, hidden1_size=784, hidden2_size=784, output_size=10)

x_test = np.array(np.genfromtxt('test_img.csv', dtype=int, delimiter=','))
y_test = np.genfromtxt('test_labels.csv', dtype=int, delimiter=',')
x_test = x_test.reshape(-1,784)
y_test = _change_one_hot_label(y_test)


W1 = np.load(r'params/W1.npy')
W2 = np.load(r'params/W2.npy')
W3 = np.load(r'params/W3.npy')
b1 = np.load(r'params/b1.npy')
b2 = np.load(r'params/b2.npy')
b3 = np.load(r'params/b3.npy')

network.set_model(W1, b1, W2, b2, W3, b3)
test_acc = network.accuracy(x_test, y_test)
print(test_acc)

def predict(x):

    y_pr = network.predict(x)

    return y_pr

print(predict(x_test[0]))
print(y_test[0])