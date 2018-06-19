# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import math
import numpy as np
import argparse
import imutils
import cv2


# Class for point
class point:
  x = 0
  y = 0
	
# Points for Left Right Top and Bottom
plL = point()
plR = point()
plU = point()
plD = point()

# For finding midpoint
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
 
# load the image, convert it to grayscale, and blur it slightly
cam = cv2.VideoCapture(0)
_ ,image = cam.read()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
 
# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)
 
# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
 
# sort the contours from left-to-right and, then initialize the
# distance colors and reference object
(cnts, _) = contours.sort_contours(cnts)
colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),
	(255, 0, 255))
refObj = None
pixelsPerMetric = None

# loop over the contours individually
for c in cnts:
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 100:
		continue
 
	# compute the rotated bounding box of the contour
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
 
	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
 
	# compute the center of the bounding box
	cX = np.average(box[:, 0])
	cY = np.average(box[:, 1])

	# if reference object is not set then we presume
	# the origin i.e. the (0,0) of the camera frame 
	# as reference point
	if refObj is None:
		box1 = np.zeros(box.shape)
		rcX = 0
		rcY = 0
		refObj = (box1, (rcX, rcY), 27.6)

	# draw the contours on the image
	orig = image.copy()
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)
 
	# stack the reference coordinates and the object coordinates
	# to include the object center
	refCoords = np.vstack([refObj[0], refObj[1]])
	objCoords = np.vstack([box, (cX, cY)])

	# Give them values 
	plL.x = box[0][0]
	plL.y = box[0][1]

	plR.x = box[1][0]
	plR.y = box[1][1]

	plU.x = box[2][0]
	plU.y = box[2][1]

	plD.x = box[3][0]
	plD.y = box[3][1]

	# Finding Height and width
	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
	
	# unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)
 
	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)
 
	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
 
	# draw lines between the midpoints
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)
	
	# compute the Euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
 
	# if the pixels per metric has not been initialized, then
	# compute it as the ratio of pixels to supplied metric
	# (in this case, inches)
	if pixelsPerMetric is None:
		pixelsPerMetric = 27.6 # This ratio needs to be set manually

    	# compute the size of the object
	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric
 
	# draw the object sizes on the image
	cv2.putText(orig, "{:.1f}in".format(dimA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
	cv2.putText(orig, "{:.1f}in".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)

	# Finding Angle 
	rp1 = point()
	rp2 = point()
	
	Angle = 0

	if(dA>=dB):
		rp1.x = tltrX
		rp1.y = tltrY
		rp2.x = blbrX
		rp2.y = blbrY
	else:
		rp1.x = tlblX
		rp1.y = tlblY
		rp2.x = trbrX 
		rp2.y = trbrY
	
	print("( " + str(rp1.x) + " , " + 
		str(rp1.y) + ") (" + str(rp2.x) + " , " + str(rp2.y) + " ) ")
	
	gradient = (rp2.y - rp1.y)*1.0/(rp2.x - rp1.x)*1.0
	Angle = math.atan(gradient)
	Angle = Angle*57.2958

	if(Angle < 0):
	    Angle = Angle + 180
	
	# Results of calculations can be printed in console or you can
	# write them on the frame itself
	print("Angle = " + str(Angle))
	print("("+ str(plL.x) + " , " + str(plL.y) + ")")
	print("("+ str(plR.x) + " , " + str(plR.y) + ")")
	print("("+ str(plU.x) + " , " + str(plU.y) + ")")
	print("("+ str(plD.x) + " , " + str(plD.y) + ")")
	print("Height = " + str(height/27.6))
	print("width = " + str(width/27.6))
	print("Loss = " + str(abs(diag1 - diag2)))

	# loop over the original points
	for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
		# draw circles corresponding to the current points and
		cv2.circle(orig, (int(xA), int(yA)), 5, color, -1)
		# connect them with a line
		cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)
		cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)), color, 2)
 
		# compute the Euclidean distance between the coordinates,
		# and then convert the distance in pixels to distance in
		# units
		D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
		(mX, mY) = midpoint((xA, yA), (xB, yB))
		cv2.putText(orig, "{:.1f}cm".format(D), (int(mX), int(mY - 10)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
 
		# show the output image
		cv2.imshow("Image", orig)
		
		cv2.waitKey(0)
