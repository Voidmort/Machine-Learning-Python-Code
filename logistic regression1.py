#!/usr/bin/python
# coding=utf-8
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

# 逻辑函数(Logistic function)
def gfunc(z):
    return 1 / (1 + np.exp(-z))

# 构造训练集：引入了鸢尾花数据集来作为训练集
iris = load_iris()
data = iris.data
target = iris.target

# 取前一百行的第一列和第三列做特征值
X = data[0:100, [0, 2]]
y = target[0:100]

# 画出训练集的散点图
label = np.array(y)
index_0 = np.where(label == 0)
plt.scatter(X[index_0, 0], X[index_0, 1], marker='x', color='b', label='0', s=15)
index_1 = np.where(label == 1)
plt.scatter(X[index_1, 0], X[index_1, 1], marker='o', color='r', label='1', s=15)
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc='upper left')

########################################################
# 训练集构建完成后判断边界，我猜边界是一条直线
# 直线的公式：θ0 * x0 + θ1 * x1 + θ2 * x2 = 0  其中x0 = 1
# 因为这个问题里是一个二维分类，所以边界是有三个θ决定的
########################################################

# 训练集的个数m
m = 100

# 重新构建了X向量 加上了x0=1
x0 = np.full(m, 1.0)
x0 = np.vstack(x0)
x = np.column_stack((x0, X))

# 随机设置三个theta值
theta = np.random.randn(3)

# 两种终止条件
loop_max = 10000  # 最大迭代次数(防止死循环)
epsilon = 1e-3

error = np.zeros(3)
count = 0
alpha = 0.001  # 步长

while count < loop_max:
    delta = np.zeros(3)
    for i  in range(m):
        delta = delta + (gfunc(np.dot(theta, x[i])) - y[i]) * x[i]
    theta = theta - alpha * delta
    # 判断是否已收敛
    if np.linalg.norm(theta - error) < epsilon:
        finish = 1
        break
    else:
        error = theta
    count += 1

print("The number of iterations = ", count)
print(theta)

# x0 = 1
# 已经求得theta参数，给出x1的值，根据theta计算x2，画出直线
x1 = np.arange(4, 7.5, 0.5)
x2 = (- theta[0] - theta[1] * x1) / theta[2]

plt.plot(x1, x2, color='black')

plt.show()
