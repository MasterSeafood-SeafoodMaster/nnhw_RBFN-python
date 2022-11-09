import matplotlib.pyplot as plt
import math
import toolkit as tk
import numpy as np
import time


square, startPoint, endline = tk.getSquare("軌道座標點.txt")

data6D = tk.dataLoader("train6dAll.txt")

startPoint = [10, 15, 90]
startPoint[2] = 0

f, l, r, fp, lp, rp = tk.Sensor(startPoint, square, 50)


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.plot(square[:, 0], square[:, 1], color="green")

ax.plot((f[0], f[2]), (f[1], f[3]), color="green")
ax.plot((l[0], l[2]), (l[1], l[3]), color="green")
ax.plot((r[0], r[2]), (r[1], r[3]), color="green")

#ax.scatter(data6D[0:20, 0], data6D[0:20, 1], color="blue")
ax.scatter(startPoint[0], startPoint[1], color="red")
ax.scatter([fp[0], lp[0], rp[0]], [fp[1], lp[1], rp[1]], color="blue")

ax.plot(endline[:, 0], endline[:, 1],color="red")

ax.axis([-20, 40, -5, 55])
plt.show()
