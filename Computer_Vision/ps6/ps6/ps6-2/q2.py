import os
import cv2 
import numpy as np


if __name__ == '__main__':
	#loading the input imGE
	filename = "/home/akshay/Downloads/CV/ps6/spade-terminal.png"
	input_image = cv2.imread(filename)
	
	#CONVERTING TO BINARY image
	gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

	#threshloding the image using binary thresholding
	thr, image = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

	#applying two erosions
	for i in range(2):
		image = cv2.erode(image, None)

	#applying 3 dilations
	for i in range(3):
		image = cv2.dilate(image, None)
	
	#extracting the contours from the images
	contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	
	#taking the template as contour number 29, which is a non-defective contour
	#it will be shown in the write-up
	req_contour = contours[29]

	#drawing the template contour on white screen
	white_screen = np.uint8(np.full(image.shape, 255))
	white_screen = cv2.drawContours(white_screen, contours, 29, (0,0,0), -1)
	cv2.imwrite("template.png", white_screen)
	similarity = np.empty((len(contours),2))

	for i, contour in enumerate(contours):
		#iterating through all the contours and store the results returned from matchshapes 
		# and the corresponding index
		similarity[i] = [cv2.matchShapes(contour, req_contour, cv2.CONTOURS_MATCH_I2, 0), i] 

	#sort based on the the similarity and reverse it
	similarity = similarity[np.argsort(similarity[:,0])]
	similarity = similarity[::-1]

	#take the first three
	for i in range(3):
		input_image = cv2.drawContours(input_image, contours, int(similarity[i,1]), (0,0,255), -1)

	#save the image
	if cv2.imwrite("spade-terminal-output.png", input_image):
		print("Saved Successfully")
	
	cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
	cv2.imshow("threshold", input_image)
	cv2.waitKey(0)