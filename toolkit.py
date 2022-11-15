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
	
def Distance(p1, p2):
	p1=np.array(p1)
	p2=np.array(p2)
	p3=p2-p1
	p4=math.hypot(p3[0],p3[1])
	
	return p4


def dataLoader(path):
	f = open(path, 'r')
	lines = f.readlines()

	data6D = []
	for line in lines:
		l = line.replace("\n", "")
		data6D.append(l.split(" "))

	data6D = np.array(data6D, dtype=np.float)

	return data6D

def inBox(p, Box):
	xIn = p[0]>min(Box[0][0], Box[1][0]) and p[0] < max(Box[0][0], Box[1][0])
	yIn = p[1]>min(Box[0][1], Box[1][1]) and p[1] < max(Box[0][1], Box[1][1])

	return xIn and yIn

def sameDir(myPos, frl, p):
	frl = np.array([frl[0]-myPos[0], frl[1]-myPos[1]])
	p = np.array([p[0]-myPos[0], p[1]-myPos[1]])
	return np.dot(frl, p)>0

def getCrossPoint(l1, ls):
	x=l1[0][0]; y=l1[0][1]
	rx = l1[1][0]-l1[0][0]
	ry = l1[1][1]-l1[0][1]
	p = "None"
	if ls[0][0] == ls[1][0]: #ver
		fx = ls[0][0]
		p = [ fx, (ry*(fx-x)/rx)+y ]
		if p[1]>max(ls[0][1], ls[1][1]) or p[1]<min(ls[0][1], ls[1][1]):
			p="None"

	elif ls[0][1] == ls[1][1]: #hor
		fy = ls[0][1]
		p = [ (rx*(fy-y)/ry)+x, fy]
		if p[0]>max(ls[0][0], ls[1][0]) or p[0]<min(ls[0][0], ls[1][0]):
			p="None"
	return p

def getLine(myPos, ang, l):
	r = math.radians(ang)
	return [myPos[0]+round(l*math.cos(r)), myPos[1]+round(l*math.sin(r))]

def Distance(p1, p2):
	print(p1, p2)
	rx = abs(p1[0]-p2[0])
	ry = abs(p1[1]-p2[1])
	return math.hypot(rx, ry)

def getNearest(myPos, pl):
	D = 500
	fp = ""
	for p in pl:
		rD = Distance(myPos, p)
		if rD<D:
			D=rD
			fp = p
	return fp

def SensorV2(myPos, square, sLength):
	fpl=[]; rpl=[]; lpl=[]
	myAngle = myPos[2]
	myPosi = [myPos[0], myPos[1]]

	f = getLine(myPosi, myAngle, sLength)
	r = getLine(myPosi, myAngle-45, sLength)
	l = getLine(myPosi, myAngle+45, sLength)

	for i in range(len(square)-1):
		fp = getCrossPoint([myPosi, f], [square[i], square[i+1]])
		if fp!="None" and sameDir(myPosi, f, fp):
			fpl.append(fp)
		
		rp = getCrossPoint([myPosi, r], [square[i], square[i+1]])
		if rp!="None" and sameDir(myPosi, r, rp):
			rpl.append(rp)
		
		lp = getCrossPoint([myPosi, l], [square[i], square[i+1]])
		if lp!="None" and sameDir(myPosi, l, lp):
			lpl.append(lp)
		
	return f, r, l,getNearest(myPosi, fpl), getNearest(myPosi, rpl), getNearest(myPosi, lpl) # 
	