#!/usr/bin/python
# coding=utf-8
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 构造训练集
# x 特征值
# y 实际结果
x1 = np.arange(0, 50, 1) + np.random.randn(50)
m = len(x1)
x0 = np.full(m, 1.0)
x = np.vstack([x0, x1]).T
y = x1/2 + np.random.randn(m) -5

# 两种终止条件
loop_max = 10000  # 最大迭代次数(防止死循环)
epsilon = 1e-4

# 初始化权值
np.random.seed(0)
theta = np.random.randn(2)

alpha = 0.002  # 步长(注意取值过大会导致振荡即不收敛,过小收敛速度变慢) 大于0.002会不收敛
error = np.zeros(2)
count = 0  # 循环次数

while count < loop_max:
    count += 1
    delta = np.zeros(2)
    for i in range(m):
        delta = delta + (np.dot(theta, x[i]) - y[i]) * x[i]/m
    theta = theta - alpha * delta

    # 判断是否已收敛
    if np.linalg.norm(theta - error) < epsilon: # np.linalg.norm 求范类：平方和，开方
        break
    else:
        error = theta
    print(theta)

print(theta,count)
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x1, y)
print('intercept = %s slope = %s' % (intercept, slope))
plt.plot(x1, y, 'g*')
plt.plot(x, theta[1] * x + theta[0], 'r')
plt.plot(x, slope * x + intercept, 'b')
plt.show()
