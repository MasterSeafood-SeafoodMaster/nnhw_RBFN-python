import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

from mlp.layers import Dense, Sigmoid
from mlp.lossfun import mse, mse_prime
from mlp.toolkit import dataLoader, train, predict, minMax, minMax_prime

import toolkit as tkit

import tkinter as tk

fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
ax.axis([-20, 40, -5, 55])

class MyApp(tk.Tk):
	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):
		#self.canvs = FigureCanvasTkAgg(fig, master=self)
		
		self.configure(background='white')

		self.Path = tk.StringVar()
		self.Classes = tk.StringVar()
		self.Epochs = tk.StringVar()
		self.Learning_rate = tk.StringVar()
		self.Model_shape = tk.StringVar()
		self.Log = tk.StringVar()

		self.title('hello world')
		self.geometry('720x720')
		self.setLabel("Path", (20, 360))
		self.setLabel("Classes", (20, 400))
		self.setLabel("Epochs", (20, 440))
		self.setLabel("Learning_rate", (20, 480))
		self.setLabel("Model_shape", (20, 520))
		self.logLabel = tk.Label(self, text="Log", font=("MV Boli", 16), bg="white", textvariable=self.Log)
		self.logLabel.place(x=20, y=670)


		self.setEntry(self.Path, (180, 360), "./train4dAll.txt", 30)
		self.setEntry(self.Classes, (180, 400), "1, 2", 10)
		self.setEntry(self.Epochs, (180, 440), "50", 10)
		self.setEntry(self.Learning_rate, (180, 480), "0.01", 10)
		self.setEntry(self.Model_shape, (180, 520), "3, 64, 1", 10)

		self.setButton("Train!", (20, 600), self.Train)
		#self.setButton("Draw_line!", (120, 600), self.Train)

	def setLabel(self, text, pos):
		label = tk.Label(self, text=text, font=("MV Boli", 16), bg="white")
		label.place(x=pos[0], y=pos[1])

	def setEntry(self, re, pos, default, width):
		entry = tk.Entry(self, font=("MV Boli", 16), textvariable=re, width=width)
		entry.insert(-1, default)
		entry.place(x=pos[0], y=pos[1])

	def setButton(self, text, pos, fun):
		button = tk.Button(self, text=text, font=("MV Boli", 16), command=fun)
		button.place(x=pos[0], y=pos[1])

	def Train(self):
		self.i = 0
		self.dataPath = self.Path.get()

		self.epochs = int(self.Epochs.get())
		
		self.learning_rate = float(self.Learning_rate.get())
		self.nn_shape = self.Model_shape.get().replace(" ", "").split(",")
		for i in range(len(self.nn_shape)): self.nn_shape[i] = int(self.nn_shape[i])

		self.datasetArray = np.loadtxt(self.dataPath, dtype=float)
		self.dL, self.eL = self.datasetArray.shape
		self.X = self.datasetArray[:, 0:3]
		self.Y = self.datasetArray[:, 3]

		self.Y = minMax(self.Y, -40, 40)
		self.X = np.reshape(self.X, (self.dL, self.eL-1, 1))
		self.Y = np.reshape(self.Y, (self.dL, 1, 1))


		self.nn = []
		for i in range(len(self.nn_shape)-1):
			self.nn.append(Dense(self.nn_shape[i], self.nn_shape[i+1]))
			self.nn.append(Sigmoid())

		train(self.nn, mse, mse_prime, self.X, self.Y, self.epochs, self.learning_rate)
		self.square, self.startPoint, self.endline = tkit.getSquare("軌道座標點.txt")
		self.myPos = self.startPoint

		self.points=[]
		while not tkit.inBox(self.myPos, self.endline):
			#ax.scatter(self.myPos[0], self.myPos[1], color="blue")
			self.f, self.r, self.l, self.fp, self.rp, self.lp = tkit.SensorV2(self.myPos, self.square, 10)

			self.fd = tkit.Distance([self.myPos[0], self.myPos[1]], self.fp)
			self.ld = tkit.Distance([self.myPos[0], self.myPos[1]], self.lp)
			self.rd = tkit.Distance([self.myPos[0], self.myPos[1]], self.rp)

			self.pred = predict(self.nn, [[self.fd], [self.rd], [self.ld]])

			self.pred = minMax_prime(self.pred, -40, 40)
			self.myPos = tkit.nextPos(self.myPos[0], self.myPos[1], self.myPos[2], self.pred[0][0])
			self.points.append(self.myPos)

		print(self.points)
		self.draw_picture(self.points)

	def run(self, data):
		global fig, ax
		ax.plot(self.square[:, 0], self.square[:, 1], color="green")
		ax.plot([self.endline[0][0], self.endline[1][0], self.endline[1][0], self.endline[0][0], self.endline[0][0]], [self.endline[0][1], self.endline[0][1], self.endline[1][1], self.endline[1][1],self.endline[0][1]],color="red")
		if self.i < len(self.points):
			ax.scatter(self.points[data][0], self.points[data][1], color="blue")
			self.i+=1


	def draw_picture(self, points):
		#global fig, ax
		ani = animation.FuncAnimation(fig, self.run, frames=len(points), interval=100, repeat=True)
		ani.save("./ani.gif", fps=10)

		self.frameCnt = len(self.points)
		self.frames = [tk.PhotoImage(file="./ani.gif",format = 'gif -index %i' %(i)) for i in range(self.frameCnt)]
		self.flabel = tk.Label(self)
		self.flabel.place(x=200, y=25)
		self.after(50, self.update, 0)
		#self.canvs.draw()
		#self.canvs.get_tk_widget().pack()

	def update(self, ind):
		self.frame = self.frames[ind]
		ind += 1
		if ind == self.frameCnt:
			ind = 0
		self.flabel.configure(background="white", image=self.frame)
		self.after(50, self.update, ind)

app = MyApp()
app.mainloop()