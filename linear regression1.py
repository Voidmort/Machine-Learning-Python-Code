#!/usr/bin/python
# coding=utf-8
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import math   # This will import math module
# 构造训练集
# x 特征值
# y 实际结果
x = np.arange(0, 50, 1)
m = len(x)
y = x/2 + np.random.randn(m) -5

# 终止条件
loop_max = 100000  # 最大迭代次数(防止死循环)
epsilon = 1e-4   # 精确度

alpha = 0.002  # 步长(注意取值过大会导致振荡即不收敛,过小收敛速度变慢)
count = 0  # 循环次数
finish = 0  # 终止标志
theta = np.random.randn(2)# 初始化theta
#theta = [0.5,-0.5]
temp = np.zeros(2)
error = 0

while count < loop_max:
    count+=1
    sum = np.zeros(2)
    for i in range(m):
        sum[0] = sum[0] + (theta[0] + theta[1] * x[i] - y[i])
    temp0 = theta[0] - alpha * sum[0] / m

    for i in range(m):
        sum[1] = sum[1]+ (theta[0] + theta[1] * x[i] - y[i]) * x[i]
    temp1 = theta[1] - alpha * sum[1] / m

    theta[0] = temp0
    theta[1] = temp1

    # 判断是否已收敛
    if abs((sum[1]+ sum[0] - error)) < epsilon:
        finish = 1
        break
    else:
        error = sum[1]+ sum[0]
    print('intercept = %s slope = %s' % (theta[0], theta[1]))


#slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, y)
#print('intercept = %s slope = %s' % (intercept, slope))
print('loop count = %d\n' % count, theta)
plt.plot(x, y, 'r*')
plt.plot(x, theta[1] * x + theta[0], 'g')
#plt.plot(x, slope * x + intercept, 'b')
plt.show()
