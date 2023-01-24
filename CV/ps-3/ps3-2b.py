import cv2
import numpy as np
import os

filename = "/home/akshay/Downloads/CV/ps-3-q/ps3-images/circuit.png"
original_image = cv2.imread(filename)
input_gray_image = cv2.cvtColor(original_image,code=cv2.COLOR_BGR2GRAY)

output_edge_image = input_gray_image
def nothing(x):
	pass

	
cv2.namedWindow("Trackbar")
cv2.createTrackbar('Threshold1',"Trackbar",0,255,nothing)
cv2.createTrackbar('Threshold2',"Trackbar",0,255,nothing)
cv2.createTrackbar('Aperture',"Trackbar",3,7,nothing)

while(1):
	cv2.imshow("Trackbar",output_edge_image)
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
	    break
	threshold1 = int(cv2.getTrackbarPos('threshold1','Trackbar'))
	threshold2 = int(cv2.getTrackbarPos('threshold2','Trackbar'))
	aperature = int(cv2.getTrackbarPos('Aperture','Trackbar'))
	if(aperature%2 == 0):
		aperature += 1
		output_edge_image = cv2.Canny(input_gray_image,threshold1,threshold2,aperature)
		output_edge_image = np.uint8(np.where(output_edge_image<127,255,0))
	else:
		output_edge_image = cv2.Canny(input_gray_image,threshold1,threshold2,aperature)
		output_edge_image = np.uint8(np.where(output_edge_image<127,255,0))
	
cv2.waitKey(0)
