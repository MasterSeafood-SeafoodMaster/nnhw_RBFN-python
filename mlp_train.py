import numpy as np
import matplotlib.pyplot as plt

from mlp.layers import Dense, Sigmoid
from mlp.lossfun import mse, mse_prime
from mlp.toolkit import dataLoader, train, predict, minMax

import toolkit as tk

dPath = "./train4dAll.txt"
datasetArray = np.loadtxt(dPath, dtype=float)
dL, eL = datasetArray.shape

classes = [-40, 40]
epochs = 1000
learning_rate = 0.01

nn_shape = [3, 20, 1]

X = datasetArray[:, 0:3]
Y = datasetArray[:, 3]

Y = minMax(Y, -40, 40)
print(Y)

X = np.reshape(X, (dL, eL-1, 1))
Y = np.reshape(Y, (dL, 1, 1))

print(X)

network = []
for i in range(len(nn_shape)-1):
	network.append(Dense(nn_shape[i], nn_shape[i+1]))
	network.append(Sigmoid())

#train(network, mse, mse_prime, X, Y, epochs, learning_rate)


square, startPoint, endline = tk.getSquare("軌道座標點.txt")
pred = predict(network, )

