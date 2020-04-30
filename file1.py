# FILE1: using SVR + CVXOPT
import numpy as np
import matplotlib
import csv
from cvxopt import matrix
from cvxopt import solvers
from sklearn.preprocessing import StandardScaler

# INPUT
filename = "BostonHousing.csv"
data = []
with open(filename, 'r') as csvfile:
	csvreader = csv.reader(csvfile)
	for row in csvreader:
		data.append(row)
data = data[1:]
for i in range(len(data)):
	for j in range(14):
		data[i][j] = float(data[i][j])

# Separating Training and Testing data
train_data = data[0:355]
test_data = data[355:]

# Normalization of input and output matrices
X = [0]*(len(train_data))
y = [0]*(len(train_data))
for i in range(len(train_data)):
	X[i] = train_data[i][0:13] + [1]
	y[i] = train_data[i][13]
for i in range(len(y)):
	y[i] = float(y[i])
for i in range(len(X)):
	for j in range(14):
		X[i][j] = float(X[i][j])
mu = np.mean(X)
std = np.std(X)
for i in range(len(X)):
	for j in range(14):
		X[i][j] = (X[i][j]-mu)/ std

# Computing the required matrices
ephsilon = 5.00
c = 1.00

P = []
for i in range(14):
	temp_vec = [0.0] * (14 + len(train_data))
	temp_vec[i] = 1.0
	P.append(temp_vec)
rest = [[0.0 for i in range(14 + len(train_data))] for j in range(len(train_data))]
P = P + rest

q = [0.0 for i in range(14)] + [c for i in range(len(train_data))]

n = len(train_data)
h = [ephsilon]*(3 * len(train_data))
for i in range(n):
	h[i] = h[i] + y[i]
for i in range(n):
	h[i+n] = h[i+n] - y[i]
for i in range(n):
	h[i+2*n] = h[i+2*n] - ephsilon

n = len(train_data)
G = [0]*(3*n)
for i in range(n):
	temp_vec = [0.0]*(n)
	temp_vec[i] = -1.0
	G[i] = X[i] + temp_vec
for i in range(n):
	temp_vec = [0.0]*(n)
	temp_vec[i] = -1.0
	temp = [0.0]*14
	for j in range(14):
		temp[j] = -X[i][j]
	G[i+n] = temp + temp_vec
for i in range(n):
	temp_vec = [0.0]*(n)
	temp_vec[i] = -1.0
	temp = [0.0]*14
	G[i + 2*n] = temp + temp_vec
G = np.array(G)

sol = solvers.qp(matrix(P), matrix(q), matrix(G, tc = 'd'), matrix(h))
