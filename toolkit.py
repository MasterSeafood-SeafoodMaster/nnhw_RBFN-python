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
	if D==0: return "None"
	x = (b0*c1-b1*c0)/D
	y = (a1*c0-a0*c1)/D

	xr = [line1[0], line1[2]]
	yr = [line1[1], line1[3]]
	inL1 = x>=min(xr) and x<=max(xr) and y>=min(yr) and y<=max(yr)

	#print("line1", line1)
	#print(xr, yr, inL1 and inL2)

	xr = [line2[0], line2[2]]
	yr = [line2[1], line2[3]]
	inL2 = x>=min(xr) and x<=max(xr) and y>=min(yr) and y<=max(yr)
	
	if inL1 and inL2:
		return x, y
	else:
		return "None"

def Distance(p1, p2):
	p1=np.array(p1)
	p2=np.array(p2)
	p3=p2-p1
	p4=math.hypot(p3[0],p3[1])
	
	return p4

def getNearest(myPos, p):
	dList = []
	for i in range(len(p)):
		dList.append( Distance([myPos[0], myPos[1]], p[i]) )
	#print("dList:", dList)
	return p[dList.index(min(dList))]


def Sensor(myPos, square, sLength):
	fList=[]
	lList=[]
	rList=[]
	r = math.radians(myPos[2]+90)
	rh = r/2

	for i in range(len(square)-1):
		
		l1 = [square[i][0], square[i][1], square[i+1][0], square[i+1][1]]
		forward = [myPos[0], myPos[1], myPos[0]+sLength*math.sin(r), myPos[1]-sLength*math.cos(r)]
		lelf = [myPos[0], myPos[1], myPos[0]+sLength*math.cos(r), myPos[1]+sLength*math.sin(r)]
		right = [myPos[0], myPos[1], myPos[0]-sLength*math.cos(r), myPos[1]-sLength*math.sin(r)]

		lf = [myPos[0], myPos[1], (lelf[2]+forward[2])/2, (lelf[3]+forward[3])/2]
		rf = [myPos[0], myPos[1], (right[2]+forward[2])/2, (right[3]+forward[3])/2]

		pp = get_line_cross_point(l1, forward)
		if pp != "None": fList.append(pp)
		
		pp = get_line_cross_point(l1, lf)
		if pp != "None": lList.append(pp)
		

		#print("rightrightrightrightrightright:")
		pp = get_line_cross_point(l1, rf)
		if pp != "None": rList.append(pp)
		
	
	fp = getNearest(myPos, fList)
	lp = getNearest(myPos, lList)
	rp = getNearest(myPos, rList)

	print(fp, lp, rp)
	return forward, lf, rf, fp, lp, rp

def dataLoader(path):
	f = open(path, 'r')
	lines = f.readlines()

	data6D = []
	for line in lines:
		l = line.replace("\n", "")
		data6D.append(l.split(" "))

	data6D = np.array(data6D, dtype=np.float)

	return data6D