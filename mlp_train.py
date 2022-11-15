import numpy as np
import matplotlib.pyplot as plt

from mlp.layers import Dense, Sigmoid
from mlp.lossfun import mse, mse_prime
from mlp.toolkit import dataLoader, train, predict, minMax, minMax_prime

import toolkit as tk

dPath = "./train4dAll.txt"
datasetArray = np.loadtxt(dPath, dtype=float)
dL, eL = datasetArray.shape

classes = [-40, 40]
epochs = 25
learning_rate = 0.01

nn_shape = [3, 64, 1]

X = datasetArray[:, 0:3]
Y = datasetArray[:, 3]

Y = minMax(Y, -40, 40)

X = np.reshape(X, (dL, eL-1, 1))
Y = np.reshape(Y, (dL, 1, 1))
#print(dL, eL-1, 1)

network = []
for i in range(len(nn_shape)-1):
	network.append(Dense(nn_shape[i], nn_shape[i+1]))
	network.append(Sigmoid())

train(network, mse, mse_prime, X, Y, epochs, learning_rate)

square, startPoint, endline = tk.getSquare("軌道座標點.txt")

myPos = startPoint


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.plot(square[:, 0], square[:, 1], color="green")
print(endline)
ax.plot([endline[0][0], endline[1][0], endline[1][0], endline[0][0], endline[0][0]], [endline[0][1], endline[0][1], endline[1][1], endline[1][1],endline[0][1]],color="red")

ax.axis([-20, 40, -5, 55])


while not tk.inBox(myPos, endline):

	ax.scatter(myPos[0], myPos[1], color="blue")
	f, r, l, fp, rp, lp = tk.SensorV2(myPos, square, 10)

	fd = tk.Distance([myPos[0], myPos[1]], fp)
	ld = tk.Distance([myPos[0], myPos[1]], lp)
	rd = tk.Distance([myPos[0], myPos[1]], rp)

	pred = predict(network, [[fd], [rd], [ld]])

	pred = minMax_prime(pred, -40, 40)
	myPos = tk.nextPos(myPos[0], myPos[1], myPos[2], pred[0][0])


	#ax.plot((f[0], f[2]), (f[1], f[3]), color="green")
	#ax.plot((l[0], l[2]), (l[1], l[3]), color="green")
	#ax.plot((r[0], r[2]), (r[1], r[3]), color="green")
	#ax.scatter(data6D[0:20, 0], data6D[0:20, 1], color="blue")
	
	#ax.scatter([fp[0], lp[0], rp[0]], [fp[1], lp[1], rp[1]], color="blue")

plt.show()