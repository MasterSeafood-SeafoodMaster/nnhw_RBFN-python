import matplotlib.pyplot as plt
import math
import toolkit as tk
import numpy as np
import time


square, startPoint, endline = tk.getSquare("軌道座標點.txt")

data6D = tk.dataLoader("train6dAll.txt")

startPoint = [10, 15, 56]
#startPoint[2] = 45

#f, l, r, fp, lp, rp = tk.Sensor(startPoint, square, 50)

fp, rp, lp = tk.SensorV2(startPoint, square, 10)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.plot(square[:, 0], square[:, 1], color="blue")

#ax.plot((startPoint[0], f[0]), (startPoint[1], f[1]), color="green")
#ax.plot((startPoint[0], r[0]), (startPoint[1], r[1]), color="green")
#ax.plot((startPoint[0], l[0]), (startPoint[1], l[1]), color="green")

ax.scatter(fp[0], fp[1], color="red")
ax.scatter(rp[0], rp[1], color="red")
ax.scatter(lp[0], lp[1], color="red")

ax.scatter(startPoint[0], startPoint[1], color="red")
ax.plot(endline[:, 0], endline[:, 1],color="red")

ax.axis([-20, 40, -5, 55])
plt.show()
