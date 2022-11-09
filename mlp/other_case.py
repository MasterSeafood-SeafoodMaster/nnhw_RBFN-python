import numpy as np
import matplotlib.pyplot as plt

from layers import Dense, Sigmoid
from lossfun import mse, mse_prime
from toolkit import dataLoader, train, predict

root = "./NN_HW1_DataSet/加分題/"
dataPath = "4satellite-6.txt"

classes = [1, 2, 3, 4, 5, 6]
epochs = 10000
learning_rate = 0.01

nn_shape = [4, 16, 8, 1]
datasetArray = np.loadtxt(root+dataPath, dtype=float)
dL, eL = datasetArray.shape

X = datasetArray[:, 0:4]
Y = datasetArray[:, 4]
newY = np.linspace(0, 1, len(classes))
for i in range(len(Y)):
	for j in range(len(classes)):
		if Y[i]==classes[j]:
			Y[i]=newY[j]

X = np.reshape(X, (dL, eL-1, 1))
Y = np.reshape(Y, (dL, 1, 1))


network = []
for i in range(len(nn_shape)-1):
	network.append(Dense(nn_shape[i], nn_shape[i+1]))
	network.append(Sigmoid())

train(network, mse, mse_prime, X, Y, epochs, learning_rate)

points = []
for a, b, c, d, z in datasetArray:
	pz = predict(network, [[a], [b], [c], [d]])
	points.append([a, b, c, d, pz[0,0]])
points = np.array(points)


fig = plt.figure(figsize=(12,10))
ax00 = fig.add_subplot(221, projection="3d")
ax00.title.set_text('Dataset\ndim 0 1')
ax00.scatter(datasetArray[:, 0], datasetArray[:, 1], datasetArray[:, 4], c=datasetArray[:, 4], cmap="plasma")

ax01 = fig.add_subplot(222, projection="3d")
ax01.title.set_text('Dataset\ndim 2 3')
ax01.scatter(datasetArray[:, 2], datasetArray[:, 3], datasetArray[:, 4], c=datasetArray[:, 4], cmap="plasma")

ax10 = fig.add_subplot(223, projection="3d")
ax10.title.set_text('Predict\ndim 0 1')
ax10.scatter(points[:, 0], points[:, 1], points[:, 4], c=points[:, 4], cmap="plasma")

ax11 = fig.add_subplot(224, projection="3d")
ax11.title.set_text('Predict\ndim 2 3')
ax11.scatter(points[:, 2], points[:, 3], points[:, 4], c=points[:, 4], cmap="plasma")

plt.show()