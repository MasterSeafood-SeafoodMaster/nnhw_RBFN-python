import numpy as np
def dataLoader(dataPath, classes):
	datasetArray = np.loadtxt(dataPath, dtype=float)
	dL, eL = datasetArray.shape
	X = datasetArray[:, 0:2]
	Y = datasetArray[:, 2]
	newY = np.linspace(0, 1, len(classes))
	for i in range(len(Y)):
		for j in range(len(classes)):
			if Y[i]==classes[j]:
				Y[i]=newY[j]

	X = np.reshape(X, (dL, eL-1, 1))
	Y = np.reshape(Y, (dL, 1, 1))

	return datasetArray, X, Y

def predict(network, input):
	output = input
	for layer in network:
		output = layer.forward(output)
	return output

def train(network, loss, loss_prime, x_train, y_train, epochs, learning_rate):
	for e in range(epochs):
		error = 0
		for x, y in zip(x_train, y_train):
			# forward
			output = predict(network, x)

			# error
			error += loss(y, output)

			# backward
			grad = loss_prime(y, output)
			for layer in reversed(network):
				grad = layer.backward(grad, learning_rate)

		error /= len(x_train)
		print(f"{e + 1}/{epochs}, error={error}")

def print_weight(network):
	for layer in network:
		try:
			print(layer.weights)
		except:
			print("sigmoid")


def minMax(ori, Min, Max):
	return (ori - Min) / (Max - Min)


def minMax_prime(mm, Min, Max):
	return (mm*(Max-Min))+Min


