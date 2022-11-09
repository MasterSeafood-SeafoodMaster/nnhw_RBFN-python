import numpy as np
import matplotlib.pyplot as plt

from layers import Dense, Sigmoid
from lossfun import mse, mse_prime
from toolkit import dataLoader, train, predict

dataPath = "./NN_HW1_DataSet/基本題/2Ccircle1.txt"
classes = [1, 2]
epochs = 1000
learning_rate = 0.1

nn_shape = [2, 20, 1]

datasetArray, X, Y = dataLoader(dataPath, classes)
network = []

for i in range(len(nn_shape)-1):
	network.append(Dense(nn_shape[i], nn_shape[i+1]))
	network.append(Sigmoid())



train(network, mse, mse_prime, X, Y, epochs, learning_rate)

points = []
for x, y, z in datasetArray:
	pz = predict(network, [[x], [y]])
	points.append([x, y, pz[0,0]])
points = np.array(points)

fig = plt.figure(figsize=(11,5))

ax0 = fig.add_subplot(121, projection="3d")
ax0.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="plasma")
ax1 = plt.subplot(122, projection='3d')
ax1.scatter(datasetArray[:, 0], datasetArray[:, 1], datasetArray[:, 2], c=datasetArray[:, 2], cmap="plasma")

plt.show()
