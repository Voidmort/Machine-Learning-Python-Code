#!/usr/bin/python
# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib
import scipy.optimize as opt
from sklearn.metrics import classification_report  # 这个包是评价报告


def load_data(path, transpose=True):
    data = sio.loadmat(path)
    y = data.get('y')
    y = y.reshape(y.shape[0])
    X = data.get('X')
    if transpose:
        X = np.array([im.reshape((20, 20)).T for im in X])
        X = np.array([im.reshape(400) for im in X])
    return X, y


X, y = load_data('./data/ex3data1.mat')


def plot_an_image(image):
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)), cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()

pick_one = np.random.randint(0, 5000)
plot_an_image(X[pick_one, :])
print('this should be {}'.format(y[pick_one]))


def plot_100_image(X):
    size = int(np.sqrt(X.shape[1]))
    sample_idx = np.random.choice(np.array(X.shape[0]), 100)
    sample_images = X[sample_idx, :]

    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))

    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(sample_images[10 * r + c].reshape((size, size)), cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()


plot_100_image(X)

# 准备数据

# y have 10 categories here. 1..10, they represent digit 0 as category 10 because matlab index start at 1
# I'll ditit 0, index 0 again
raw_X, raw_y = load_data('./data/ex3data1.mat')
X = np.insert(raw_X, 0, values=np.ones(raw_X.shape[0]), axis = 1) # 插入了第一列 全为1
print(X.shape)

y_matrix = []
for k in range(1, 11):
    y_matrix.append((raw_y == k).astype(int))

y_matrix = [y_matrix[-1]] + y_matrix[:-1]
y = np.array(y_matrix)
print(y.shape)
# 扩展 5000*1 到 5000*10
#     比如 y=10 -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]: ndarray
#     比如 y=1 -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]: ndarray
# print(y.shape) # (10, 5000)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 梯度就是jθ的在θ偏导
def gradient(theta, X, y):
    # @ 对应元素相乘求和
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)


def cost(theta, X, y):
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))


def regularized_cost(theta, X, y, l=1):
    theta_j1_to_n = theta[1:]
    regularized_term = (1 / (2 * len(X))) * np.power(theta_j1_to_n, 2).sum()

    return cost(theta, X, y) + regularized_term


def regularized_gradient(theta, X, y, l=1):
    theta_j1_to_n = theta[1:]
    regularized_theta = (l / len(X)) * theta_j1_to_n
    regularized_term = np.concatenate([np.array([0]), regularized_theta]) # 在theta矩阵前接一个[0]
    return gradient(theta, X, y) + regularized_term


def logistic_regression(X, y, l=1):
    theta = np.zeros(X.shape[1])
    res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y, l), method='TNC', jac=regularized_gradient, options={'disp': True})
    final_theta = res.x
    return final_theta


def predict(x, theta):
    prob = sigmoid(x @ theta)
    print(prob)
    return (prob >= 0.5).astype(int)


# print(X.shape) # (5000, 401)
# print(y[0].shape) # (5000, )
t0 = logistic_regression(X, y[0])
print(t0.shape) # (401,)
y_pred = predict(X, t0)
print(y_pred.shape) # (5000, )
print('Accuracy={}'.format(np.mean(y[0] == y_pred)))

# 训练K维模型
k_theta = np.array([logistic_regression(X, y[k]) for k in range(10)])
print(k_theta.shape) # (10, 401)

prob_matrix = sigmoid(X @ k_theta.T)
np.set_printoptions(suppress=True) # 科学计数法表示
print(prob_matrix.shape) # (5000, 10)

y_pred = np.argmax(prob_matrix, axis=1)
print(y_pred.shape) # (5000,)

y_answer = raw_y.copy()
y_answer[y_answer==10] = 0

print(classification_report(y_answer, y_pred))


def load_weight(path):
    data = sio.loadmat(path)
    return data['Theta1'], data['Theta2']


theta1, theta2 = load_weight('./data/ex3weights.mat')
X, y = load_data('./data/ex3data1.mat', transpose=False)

X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)  # intercept

a1 = X

z2 = a1 @ theta1.T  # (5000, 401) @ (25,401).T = (5000, 25)
print(z2.shape)

z2 = np.insert(z2, 0, values=np.ones(z2.shape[0]), axis=1)

a2 = sigmoid(z2)

z3 = a2 @ theta2.T

a3 = sigmoid(z3)
print(np.argmax(a3[0,:]) + 1)
y_pred = np.argmax(a3, axis=1) + 1  # numpy is 0 base index, +1 for matlab convention，返回沿轴axis最大值的索引，axis=1代表行
print(y, y_pred)
print(classification_report(y, y_pred))
