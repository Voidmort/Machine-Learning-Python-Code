#!/usr/bin/python
# coding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# 在一个简单的二位数据集中 SVM中不同的C处理结果
raw_data = loadmat('data/ex6data1.mat')
print(raw_data.keys())

data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
data['y'] = raw_data['y']

positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))

ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')
ax.legend()
plt.show()

svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)
print(svc)

# 首先看下C=1的结果
svc.fit(data[['X1', 'X2']], data['y'])
score = svc.score(data[['X1', 'X2']], data['y'])
print(score)

# 当C=100的时候
svc2 = svm.LinearSVC(C=100, loss='hinge', max_iter=1000)
svc2.fit(data[['X1', 'X2']], data['y'])
score2 = svc2.score(data[['X1', 'X2']], data['y'])
print(score2) # 每次执行的结果可能不同

data['SVM 1 Confidence'] = svc.decision_function(data[['X1', 'X2']])

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM 1 Confidence'], cmap='seismic')
ax.set_title('SVM (C=1) Decision Confidence')
plt.show()

data['SVM 2 Confidence'] = svc2.decision_function(data[['X1', 'X2']])

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM 2 Confidence'], cmap='seismic')
ax.set_title('SVM (C=100) Decision Confidence')
plt.show()


# 核函数
def gaussian_kernel(x1, x2, sigma):
    return np.exp(-(np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2))))


x1 = np.array([1.0, 2.0, 1.0])
x2 = np.array([0.0, 4.0, -1.0])
sigma = 2

gaussian_kernel(x1, x2, sigma)

# 0.32465246735834974

raw_data = loadmat('data/ex6data2.mat')
data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
data['y'] = raw_data['y']

positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['X1'], positive['X2'], s=30, marker='x', label='Positive')
ax.scatter(negative['X1'], negative['X2'], s=30, marker='o', label='Negative')
ax.legend()
plt.show()

svc = svm.SVC(C=100, gamma=10, probability=True)
print(svc)

svc.fit(data[['X1', 'X2']], data['y'])
svc.score(data[['X1', 'X2']], data['y'])
data['Probability'] = svc.predict_proba(data[['X1', 'X2']])[:,0]
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(data['X1'], data['X2'], s=30, c=data['Probability'], cmap='Reds')
plt.show()

# 搜索最佳参数
raw_data = loadmat('data/ex6data3.mat')
X = raw_data['X']
Xval = raw_data['Xval']
y = raw_data['y'].ravel()
yval = raw_data['yval']. ravel()

C_values = [0.001, 0.003, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

best_score = 0
best_params = {'C': None, 'gamma':None}

for C in C_values:
    for gamma in gamma_values:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(X, y)
        score = svc.score(Xval, yval)

        if score > best_score:
            best_score = score
            best_params['C'] = C
            best_params['gamma'] = gamma

print(best_params, best_score)

# 垃圾邮件过滤
mat_tr = loadmat('data/spamTrain.mat')
X, y = mat_tr.get('X'), mat_tr.get('y').ravel()
print(X.shape, y.shape)  # ((4000, 1899), (4000,))

mat_test = loadmat('data/spamTest.mat')
test_X, test_y = mat_test.get('Xtest'), mat_test.get('ytest').ravel()
print(test_X.shape, test_y.shape)  # ((1000, 1899), (1000,))

svc = svm.SVC()
svc.fit(X, y)
pred = svc.predict(test_X)
print(metrics.classification_report(test_y, pred))

svc = svm.SVC(C=100)
svc.fit(X, y)
pred = svc.predict(test_X)
print(metrics.classification_report(test_y, pred))

# 如果是逻辑回归呢？
logit = LogisticRegression()
logit.fit(X, y)
pred = logit.predict(test_X)
print(metrics.classification_report(test_y, pred))
