import numpy as np
import math

def getSquare(path):
	f = open(path, 'r')
	lines = f.readlines()
	car_pos = lines.pop(0).replace("\n", "").split(",")
	start = lines.pop(0).replace("\n", "").split(",")
	end = lines.pop(0).replace("\n", "").split(",")

	square = []
	for line in lines:
		l = line.replace("\n", "")
		square.append(l.split(","))

	square = np.array(square, dtype=np.float)
	car_pos = np.array(car_pos, dtype=np.float)
	endline = np.array([start,end], dtype=np.float)

	return square, car_pos, endline

def nextPos(x, y, ang, theta):
	r_ang = math.radians(ang)
	r_theta = math.radians(theta)

	nx = x + math.cos( math.radians(ang+theta) ) + (math.sin( math.radians(theta) )*math.sin( math.radians(ang) ))
	ny = y + math.sin( math.radians(ang+theta) ) - (math.sin( math.radians(theta) )*math.cos( math.radians(ang) ))

	nang = math.radians(ang) - math.asin( (math.sin( math.radians(theta) )*2)/6 )
	nang = nang*(180/math.pi)

	#print(nx-x, ny-y, nang)

	return np.array([nx, ny, nang], dtype=np.float)


def calc_abc_from_line_2d(x0, y0, x1, y1):
	a = y0-y1
	b = x1-x0
	c = x0*y1-x1*y0
	return a, b, c


def get_line_cross_point(line1, line2):
	a0, b0, c0 = calc_abc_from_line_2d(*line1)
	a1, b1, c1 = calc_abc_from_line_2d(*line2)
	D = a0*b1-a1*b0
	if D==0:
		return "None"
	x = (b0*c1-b1*c0)/D
	y = (a1*c0-a0*c1)/D
	return x, y
