import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from layers import Dense, Sigmoid
from lossfun import mse, mse_prime
from toolkit import dataLoader, predict, print_weight, computeAcc

fig = plt.figure(figsize=(6,3))
ax0 = fig.add_subplot(121, projection="3d")
ax1 = fig.add_subplot(122, projection='3d')

class MyApp(tk.Tk):
	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):
		self.canvs = FigureCanvasTkAgg(fig, self)
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


		self.setEntry(self.Path, (180, 360), "./NN_HW1_DataSet/基本題/2Ccircle1.txt", 30)
		self.setEntry(self.Classes, (180, 400), "1, 2", 10)
		self.setEntry(self.Epochs, (180, 440), "50", 10)
		self.setEntry(self.Learning_rate, (180, 480), "0.1", 10)
		self.setEntry(self.Model_shape, (180, 520), "2, 20, 1", 10)

		self.setButton("Train!", (20, 600), self.Train)
		self.setButton("Draw_line!", (120, 600), self.draw_line)

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
		self.dataPath = self.Path.get()
		self.classes = self.Classes.get().replace(" ", "").split(",")
		for i in range(len(self.classes)): self.classes[i] = float(self.classes[i])
		self.epochs = int(self.Epochs.get())
		self.learning_rate = float(self.Learning_rate.get())
		self.nn_shape = self.Model_shape.get().replace(" ", "").split(",")
		for i in range(len(self.nn_shape)): self.nn_shape[i] = int(self.nn_shape[i])

		self.datasetArray, self.X, self.Y = dataLoader(self.dataPath, self.classes)
		self.nn = []
		for i in range(len(self.nn_shape)-1):
			self.nn.append(Dense(self.nn_shape[i], self.nn_shape[i+1]))
			self.nn.append(Sigmoid())

		for e in range(self.epochs):
			self.error = 0
			for x, y in zip(self.X, self.Y):
				self.output = predict(self.nn, x)
				self.error += mse(y, self.output)
				self.grad = mse_prime(y, self.output)
				for layer in reversed(self.nn):
					self.grad = layer.backward(self.grad, self.learning_rate)
			self.error /= len(self.X)
			print(f"{e + 1}/{self.epochs}, mse={self.error}")


		print_weight(self.nn)
		print("acc:", computeAcc(self.nn, self.X, self.Y))
		self.Log.set(f"Success!, mse={round(self.error, 4)}")
		self.points = []
		for x, y, z in self.datasetArray:
			self.pz = predict(self.nn, [[x], [y]])
			self.points.append([x, y, self.pz[0,0]])
		self.points = np.array(self.points)

		self.draw_picture(self.points)
			
			
	def draw_picture(self, points):
		global fig, ax0, ax1
		ax0.cla()
		ax1.cla()		

		ax0.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c=self.points[:, 2], cmap="plasma")
		ax1.scatter(self.datasetArray[:, 0], self.datasetArray[:, 1], self.datasetArray[:, 2], c=self.datasetArray[:, 2], cmap="plasma")
		self.canvs.draw()
		self.canvs.get_tk_widget().pack()


	def draw_line(self):
		global fig, ax0, ax1
		self.points = []
		for x in np.linspace(min(self.datasetArray[:, 0]), max(self.datasetArray[:, 0]), 100):
			for y in np.linspace(min(self.datasetArray[:, 1]), max(self.datasetArray[:, 1]), 100):
				self.pz = round(predict(self.nn, [[x], [y]])[0,0], 1)
				if self.pz == 0.5:
					self.points.append([x, y, self.pz])

		
		self.points = np.array(self.points)
		self.points = np.sort(self.points, axis=0)
		print(self.points)


		ax0.plot(self.points[:, 0], self.points[:, 1], self.points[:, 2], c=(0, 0, 0))
		self.canvs.draw()
		self.canvs.get_tk_widget().pack()



app = MyApp()
app.mainloop()
