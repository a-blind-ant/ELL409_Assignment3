# File2: SVR using customized solvers (from sklearn)
import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
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
train_data = data
test_data = data

# Finding the required matrices
X = [0]*(len(train_data))
y = [0]*(len(train_data))
for i in range(len(train_data)):
	X[i] = train_data[i][0:13]
	y[i] = train_data[i][13]

c = 1.0

# Normalization:
scale_X = StandardScaler()
X = scale_X.fit_transform(X)


# Regression
c_ls = [1.0/500, 5.0/500, 10.0/500, 50.0/500, 100.0/500, 500.0/500, 1000.0/500]
ram = int(input())
regressor = SVR(kernel = 'poly', degree = 2, C = c_ls[ram]*500, gamma = 'auto', epsilon = 5.0)
regressor.fit(X, y)

X = [0]*(len(test_data))
y = [0]*(len(test_data))
for i in range(len(test_data)):
	X[i] = test_data[i][0:13]
	y[i] = test_data[i][13]
scale_X = StandardScaler()
X = scale_X.fit_transform(X)	


# Prediction
y_pred = regressor.predict(X)
