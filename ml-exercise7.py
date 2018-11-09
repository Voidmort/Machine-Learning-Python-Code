#!/usr/bin/python
# coding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat


# K-means 聚类
def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)

    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i, :] - centroids[j, :]) ** 2)

            if dist < min_dist:
                min_dist = dist

                idx[i] = j

    return idx


# 数据可视化
data = loadmat('data/ex7data2.mat')
X = data['X']
data2 = pd.DataFrame(data.get('X'), columns=['X1', 'X2'])
print(data2.head())
sb.set(context="notebook", style="white")
sb.lmplot('X1', 'X2', data=data2, fit_reg=False)
plt.show()

initial_centroids = initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

idx = find_closest_centroids(X, initial_centroids)

print(idx[: 3])


def compute_centroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k, n))

    for i in range(k):
        indices = np.where(idx == i)
        centroids[i, :] = (np.sum(X[indices, :], axis=1) / len(indices[0])).ravel()

    return centroids


print(compute_centroids(X, idx, 3))

centroids_trace = np.empty(shape=[0, 2])


def run_k_means(X, initial_centroids, max_iters):
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids
    global centroids_trace
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)
        #centroids_trace = np.append(centroids_trace, centroids, axis=0)

    return idx, centroids


def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)

    for i in range(k):
        centroids[i, :] = X[idx[i], :]

    return centroids


initial_centroids = init_centroids(X, 3)
idx, centroids = run_k_means(X, initial_centroids, 10)

cluster1 = X[np.where(idx == 0)[0], :]
cluster2 = X[np.where(idx == 1)[0], :]
cluster3 = X[np.where(idx == 2)[0], :]

fig, ax = plt.subplots(figsize=(12, 8))

ax.scatter(cluster1[:, 0], cluster1[:, 1], s=30, color='r', label='Cluster 1')
ax.scatter(cluster2[:, 0], cluster2[:, 1], s=30, color='g', label='Cluster 2')
ax.scatter(cluster3[:, 0], cluster3[:, 1], s=30, color='b', label='Cluster 3')
ax.legend()

x = centroids_trace[:, 0]
y = centroids_trace[:, 1]
ax.scatter(x, y, color='black', s=50, zorder=2)
plt.show()


# Image compression with K-means
image_data = loadmat('data/bird_small.mat')
print(image_data)
A = image_data['A']
print(A.shape)

# 数据预处理
# normalize value ranges
A = A / 255

# reshape the array
print((A.shape[0] * A.shape[1], A.shape[2]))
X = np.reshape(A, (128*128, 3))
print(X.shape)

# randomly initalize the centroids
initial_centroids = init_centroids(X, 16)

# run the algorithm
idx, centroids = run_k_means(X, initial_centroids, 10)

# gor the closet centroids one last time
idx = find_closest_centroids(X, centroids)

# map each poxel to the centroid value
X_recovered = centroids[idx.astype(int), :]
print(X_recovered.shape)

# reshape to the original dimensions
X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))

# 用scikit-learn来实现K-means
from sklearn.cluster import KMeans  # 导入kmeans库

model = KMeans(n_clusters=16, n_init=100, n_jobs=1)
model.fit(X)
centroids = model.cluster_centers_
print(centroids.shape)
C = model.predict(X)
print(C.shape)
print(centroids[C].shape)

compressed_pic = centroids[C].reshape((128, 128, 3))

fig, ax = plt.subplots(1, 3)
ax[0].imshow(A)
ax[1].imshow(X_recovered)
ax[2].imshow(compressed_pic)
plt.show()


data = loadmat('data/ex7data1.mat')
X = data['X']


def pca(X):
    # normalize the feature
    X = (X - X.mean()) / X.std()

    # compute the covariance matrix
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]

    # perform SVD
    U, S, V = np.linalg.svd(cov)

    return U, S, V


U, S, V = pca(X)
print(U, S, V)


def project_data(X, U, k):
    U_reduced = U[:, :k]
    return np.dot(X, U_reduced)


Z = project_data(X, U, 1)


def recover_data(Z, U, k):
    U_reduced = U[:, :k]
    return np.dot(Z, U_reduced.T)


X_recovered = recover_data(Z, U, 1)

fig, ax = plt.subplots(1, 2)
ax[0].scatter(X[:, 0], X[:, 1])
ax[1].scatter(list(X_recovered[:, 0]), list(X_recovered[:, 1]))
plt.show()


def plot_n_image(X, n):
    """ plot first n images
    n has to be a square number
    """
    pic_size = int(np.sqrt(X.shape[1]))
    grid_size = int(np.sqrt(n))
    first_n_images = X[:n, :]

    fig, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size,
                                    sharey=True, sharex=True, figsize=(8, 8))
    for r in range(grid_size):
        for c in range(grid_size):
            ax_array[r, c].imshow(first_n_images[grid_size * r + c].reshape((pic_size, pic_size)).T, cmap=plt.cm.gray)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))


faces = loadmat('data/ex7faces.mat')
X = faces['X']
print(X.shape)
plot_n_image(X, 100)
face1 = np.reshape(X[1, :], (32, 32)).T

U, S, V = pca(X)
Z = project_data(X, U, 100)
print(Z.shape)
X_recovered = recover_data(Z, U, 100)
face2 = np.reshape(X_recovered[1,:], (32, 32)).T

fig, ax = plt.subplots(1, 2)
ax[0].imshow(face1, cmap=plt.cm.gray)
ax[1].imshow(face2, cmap=plt.cm.gray)
plt.show()
# 计算平均均方差误差与训练集方差的比例
print(np.sum(S[:100]) / np.sum(S))  # 0.9434273519364477


