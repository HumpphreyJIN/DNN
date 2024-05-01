import numpy as np
import matplotlib.pylab as plt
from Net import Net
from Net import _change_one_hot_label


# 读入数据
x_train = np.array(np.genfromtxt('train_img.csv', dtype=int, delimiter=','))
y_train = np.genfromtxt('train_labels.csv', dtype=int, delimiter=',')
x_test = np.array(np.genfromtxt('test_img.csv', dtype=int, delimiter=','))
y_test = np.genfromtxt('test_labels.csv', dtype=int, delimiter=',')

x_train, x_test = x_train / 255.0, x_test / 255.0 #归一化
x_train = x_train.reshape(-1,784)
x_test = x_test.reshape(-1,784)
y_train = _change_one_hot_label(y_train) #标签独热
y_test = _change_one_hot_label(y_test) #标签独热


network = Net(input_size=784, hidden1_size=784, hidden2_size=784, output_size=10)

#超参数设置
iters_num = 15000
train_size = x_train.shape[0]
batch_size = 1024
learning_rate = 0.05

train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

#训练
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    # 梯度
    grad = network.gradient(x_batch, y_batch)

    # 更新
    for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
        network.params[key] -= learning_rate * grad[key]

    train_loss = network.loss(x_batch, y_batch)
    train_loss_list.append(train_loss)
    test_loss = network.loss(x_test[:20], y_test[:20])
    test_loss_list.append(test_loss)


    #每一个epoch打印训练和测试的准确率
    if i % iter_per_epoch == 0:

        train_acc = network.accuracy(x_train, y_train)
        test_acc = network.accuracy(x_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f'{i+1}: 训练损失 {train_loss}, 测试损失 {test_loss},, 训练集准确率 {train_acc} 测试集准确率 {test_acc}')

#保存参数供查询
network.save_parameters(r'params')

# 绘制 loss 曲线

plt.title('Tra Loss Function Curve')
plt.ylabel('Loss')
plt.plot(train_loss_list, label="$Loss_{train}$")
plt.legend()
plt.show()

plt.title('Test Loss Function Curve')
plt.ylabel('Loss')
plt.plot(test_loss_list, label="$Loss_{test}$")
plt.legend()
plt.show()

plt.title('Loss Function Curve')
plt.ylabel('Loss')  #
plt.plot(train_loss_list, label="$Loss_{train}$")
plt.plot(test_loss_list, label="$Loss_{test}$")
plt.legend()
plt.show()

# 绘制 Accuracy 曲线
plt.title('Acc Curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(train_acc_list, label="$train_{acc}$")
plt.plot(test_acc_list, label="$test_{acc}$")
plt.legend()
plt.show()







