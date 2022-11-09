import matplotlib.pyplot as plt
import math
import toolkit as tk
import numpy as np
import time

square, startPoint, endline = tk.getSquare("軌道座標點.txt")

f = open("train4dAll.txt", 'r')
lines = f.readlines()

data4D = []
for line in lines:
	l = line.replace("\n", "")
	data4D.append(l.split(" "))

data4D = np.array(data4D, dtype=np.float)
print(endline)


path = []
car_pos = startPoint.copy()
for data in data4D:
	car_pos = tk.nextPos(car_pos[0], car_pos[1], car_pos[2], data[3])


	path.append([car_pos[0], car_pos[1]])
path = np.array(path, dtype=np.float)


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)

ax.plot(square[:, 0], square[:, 1], color="green")
ax.plot(path[:, 0], path[:, 1], color="blue")
ax.plot(endline[:, 0], endline[:, 1],color="red")

"""
def init():
	#ax.set_zlim(0,10)
	ax.plot(square[:, 0], square[:, 1], '-b')

def run(data):
	ax.cla()
	init()
	ax.scatter(5, 5)
	pass

ani = animation.FuncAnimation(fig, run, frames=1, interval=500, init_func=init)
"""
plt.show()
