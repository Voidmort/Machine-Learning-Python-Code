#!/usr/bin/python
# coding=utf-8
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

sns.set(context="notebook", style="white")

# Y是包含从1到5的等级的（数量的电影x数量的用户）数组.R是包含指示用户是否给电影评分的二进制值的“指示符”数组。
movies_mat = sio.loadmat('./data/ex8_movies.mat');
Y, R = movies_mat.get('Y'), movies_mat.get('R')
print(Y.shape, R.shape)
# (1682, 943) (1682, 943)

m, u = Y.shape
# m: how many movies
# u: how many users
n = 10
# how many features for a movie

param_mat = sio.loadmat('./data/ex8_movieParams.mat')
theta, X = param_mat.get('Theta'), param_mat.get('X')
print(theta.shape, X.shape)
# (943, 10) (1682, 10)


def serialize(X, theta):
    # serialize 2 matrix
    # X(move, feature), (1682, 10): movie features
    # theta (user, feature), (943, 10): user preference
    # 1682*10 + 943*10 = (26250,)
    return np.concatenate((X.ravel(), theta.ravel()))


def deserialize(param, n_movie, n_user, n_featuers):
    # into ndarray of X(1682, 10), theta(943, 10)
    return param[:n_movie * n_featuers].reshape(n_movie, n_featuers),\
            param[n_movie * n_featuers:].reshape(n_user, n_featuers)


# recomendation fn
def cost(param, Y, R, n_features):
    """compute cost for every r(i, j) = 1
        arg:
            param: serialized X, theta
            Y (movie, user), (1682, 943): (movie, user) rating
            R (movie, user), (1682, 943): (movie, user) has rating
    """
    # theta (user, feat)
    # X(movie, feature), (1682, 10): movie features
    n_movie, n_user = Y.shape
    X, theta = deserialize(param, n_movie, n_user, n_features)
    inner = np.multiply(X @ theta.T - Y, R)
    return np.power(inner, 2).sum() / 2


def gradient(param, Y, R, n_features):
    # theta (user, feature), (943, 10): user preference
    # X(movie, feature), (1682, 10): movie features
    n_movies, n_user = Y.shape
    X, theta = deserialize(param, n_movies, n_user, n_features)

    inner = np.multiply(X @ theta.T - Y, R)  # (1682, 943)

    # X_grad (1682, 10)
    X_grad = inner @ theta

    # theta_grad (943, 10)
    theta_grad = inner.T @ X

    # roll them together and return
    return serialize(X_grad, theta_grad)


def regularized_cost(param, Y, R, n_features, l=1):
    reg_term = np.power(param, 2).sum() * (1/2)
    return cost(param, Y, R, n_features) + reg_term


def regularized_gradient(param, Y, R, n_features, l=1):
    grad = gradient(param, Y, R, n_features)
    reg_term = l * param
    return grad + reg_term


# 按照练习中给出计算结果为22
users = 4
movies = 5
features = 3

X_sub = X[:movies, :features]
theta_sub = theta[:users, :features]
Y_sub = Y[:movies, :users]
R_sub = R[:movies, :users]

param_sub = serialize(X_sub, theta_sub)
c = cost(param_sub, Y_sub, R_sub, features)
print(c)  # 22.224603725685675

# total readl params
param = serialize(X, theta)
# total cost
total_cost = cost(param, Y, R, 10)
print(total_cost)  # 27918.64012454421

n_movie, n_user = Y.shape
X_grad, theta_grad = deserialize(gradient(param, Y, R, 10),
                                n_movie, n_user, 10)

assert X_grad.shape == X.shape
assert theta_grad.shape == theta.shape

# regularized cost
# in the ex8_confi.m, lambda = 1.5, and it's using sub data set
reg_cost = regularized_cost(param_sub, Y_sub, R_sub, features, l=1.5)
print(reg_cost)  # 28.304238738078038
# total regularized cost
total_cost = regularized_cost(param, Y, R, 10, l=1)
print(total_cost)  # 32520.682450229557

n_movie, n_user = Y.shape

X_grad, theta_grad = deserialize(regularized_gradient(param, Y, R, 10),
                                                      n_movie, n_user, 10)

assert X_grad.shape == X.shape
assert theta_grad.shape == theta.shape


# parse movie_id.txt
movie_list = []
with open('./data/movie_ids.txt', encoding='latin-1') as f:
    for line in f:
        tokens = line.strip().split(' ')
        movie_list.append(' '.join(tokens[1:]))

movie_list = np.array(movie_list)

# reproduce my ratings
ratings = np.zeros(1682)
ratings[0] = 4
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[97] = 2
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5

# prepare data
# now I become user 0
Y, R = movies_mat.get('Y'), movies_mat.get('R')
Y = np.insert(Y, 0, ratings, axis=1)
R = np.insert(R, 0, ratings != 0, axis=1)
print(Y.shape)  # (1682, 944)
print(R.shape)  # (1682, 944)

n_features = 50
n_movie, n_user = Y.shape
l = 10

# 转换为正态分布
X = np.random.standard_normal((n_movie, n_features))
theta = np.random.standard_normal((n_user, n_features))

print(X.shape, theta.shape)  # (1682, 50) (944, 50)

param = serialize(X, theta)

# normalized ratings
Y_norm = Y - Y.mean()
print(Y_norm.mean())  # 4.6862111343939375e-17

# training
import scipy.optimize as opt
res = opt.minimize(fun=regularized_cost,
                   x0=param,
                   args=(Y_norm, R, n_features, l),
                   method='TNC',
                   jac=regularized_gradient)

print(res)

X_trained, theta_trained = deserialize(res.x, n_movie, n_user, n_features)
print(X_trained.shape, theta_trained.shape)

prediction = X_trained @ theta_trained.T
my_preds = prediction[:, 0] + Y.mean()

idx = np.argsort(my_preds)[::-1]  # descending order
print(idx.shape)

# top ten idx
my_preds[idx][:10]

for m in movie_list[idx][:10]:
    print(m)