#!/usr/bin/python
# coding=utf-8
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

filename='E:\pythonlearn\\testSet.txt' #文件目录

class Logistic(object):
    # 初始化要计算的thate的值
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # 随机设置thrta_m个theta值
        self.theta_m = 3
        self.theta = np.random.randn(3)

    # 逻辑函数
    def gfunc(self, z):
        return 1 / (1 + np.exp(-z))

    # 预测函数
    def hfunc(self, x, thate):
        z = np.dot(thate, x)
        return self.gfunc(z)

    def train(self):
        x = self.x
        y = self.y
        theta = self.theta
        self.m = len(x)

        # 两种终止条件
        loop_max = 10000  # 最大迭代次数(防止死循环)
        epsilon = 1e-3
        self.finish = 0 # 终止标志
        error = np.zeros(self.theta_m)
        count = 0
        self.alpha = 0.001  # 步长
        self.loss = []

        while count < loop_max:
            delta = np.zeros(3)
            for i in range(self.m):
                h = self.hfunc(theta, x[i])
                delta = delta + (h - y[i]) * x[i]
            theta = theta - self.alpha * delta

            # 算出一个theta计算一下loss
            self.loss.append(self.compute_loss(theta))

            # 判断是否已收敛
            if np.linalg.norm(theta - error) < epsilon:
                self.finish = 1
                break
            else:
                error = theta
            count += 1
        return theta, count

    def compute_loss(self, theta):
        x = self.x
        y = self.y
        num_train = x.shape[0]
        loss = 0
        for i in range(self.m):
            h = self.hfunc(x[i], theta)
            loss += (y[i] * np.log(h) + (1 - y[i]) * np.log((1 - h)))

        loss = -loss / self.m
        return loss

    def plotloss(self):
        plt.plot(self.loss)
        plt.show()

    # x1 = 画线的范围
    def plotlogistic(self, x, y, theta, x1):
        # 画出训练集的散点图
        index_0 = np.where(y == 0)
        index_1 = np.where(y == 1)

        plt.scatter(x[index_0, 1], x[index_0, 2], marker='x', color='b', label='0', s=15)
        plt.scatter(x[index_1, 1], x[index_1, 2], marker='o', color='r', label='1', s=15)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend(loc='upper left')

        if(self.finish == 1):
            # x0 = 1
            # 已经求得theta参数，给出x1的值，根据theta计算x2，画出直线
            #x1 = np.array(-4, 4)
            x2 = (- theta[0] - theta[1] * x1) / theta[2]
            plt.plot(x1, x2, color='black')
        plt.show()

def loadDataSet():   #读取数据（这里只有两个特征）
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])   #前面的1，表示方程的常量。比如两个特征X1,X2，共需要三个参数，W1+W2*X1+W3*X2
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def main():
    x, y = loadDataSet()
    x = np.array(x)
    y = np.array(y)

    Log = Logistic(x, y)

    theta, count = Log.train()
    print("The number of iterations = ", count)
    print(theta)
    x1 = np.array([-4, 4])
    Log.plotlogistic(x, y, theta, x1)
    Log.plotloss()

if __name__=='__main__':
    main()


