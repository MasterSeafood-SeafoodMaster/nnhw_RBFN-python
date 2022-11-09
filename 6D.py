import matplotlib.pyplot as plt
import math
import toolkit as tk
import numpy as np
import time


square, startPoint, endline = tk.getSquare("軌道座標點.txt")

f = open("train6dAll.txt", 'r')
lines = f.readlines()

data6D = []
for line in lines:
	l = line.replace("\n", "")
	data6D.append(l.split(" "))

data6D = np.array(data6D, dtype=np.float)


idx=0
p=[]
for i in range(len(square)-1):
	r = math.radians(data6D[20][5])
	l1 = [square[i][0], square[i][1], square[i+1][0], square[i+1][1]]

	hor = [data6D[idx][0], data6D[idx][1], math.cos(r), math.sin(r)]
	ver = [data6D[idx][0], data6D[idx][1], math.sin(r), math.cos(r)]

	pp = tk.get_line_cross_point(l1, hor)
	if pp != "None": p.append(pp)
	pp = tk.get_line_cross_point(l1, ver)
	if pp != "None": p.append(pp)	
		

print(p)
p = np.array(p, dtype=np.float)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.plot(square[:, 0], square[:, 1], color="green")
#ax.scatter(data6D[0:20, 0], data6D[0:20, 1], color="blue")
ax.scatter(data6D[idx][0], data6D[idx][1], color="red")
ax.scatter(p[:, 0], p[:, 1], color="blue")

ax.plot(endline[:, 0], endline[:, 1],color="red")
plt.show()
