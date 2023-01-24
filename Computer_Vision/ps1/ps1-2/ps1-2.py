import cv2
import numpy as np
import os

#Taking input image from the user int the form of file name
#Taking input on whether to expose dark or bright regions
file_name = input("Please enter the location of the image:")
intensity = input("Do you want to emphasize brighter or darker regions enter b for brighter d for darker :")

#reading the input image
first_image = cv2.imread(file_name)

#printing the org image
cv2.imshow("(a) input color image", first_image)

#get the image name from the input filename
image_name = os.path.basename(file_name)
image_name = image_name.split(".",1)[0]

#converting into GRAY_SCALE image
gray_image = cv2.cvtColor(src=first_image,code=cv2.COLOR_BGR2GRAY)

#converting to binary image with a threshold of 90 for circuit and 150 for crack
if(image_name == "crack"):
	_,binary_image = cv2.threshold(gray_image,150,255,type=cv2.THRESH_BINARY)

elif(image_name == "circuit"):
	_,binary_image = cv2.threshold(gray_image,90,255,type=cv2.THRESH_BINARY)

m, n  = binary_image.shape
output = first_image
#if we want to convert dark regions to red, specifically for crack.png
if intensity == 'd':
	for i in range(m):
		for j in range(n):
			if(binary_image[i,j] == 0):
				output[i,j,2] = 255 
				output[i,j,0], output[i,j,1] = 0, 0

#if we want to convert bright regions to red, specifically for circuit.png
elif intensity == 'b':
	for i in range(m):
		for j in range(n):
			if(binary_image[i,j] == 255):
				output[i,j,2] = 255 
				output[i,j,0], output[i,j,1] = 0, 0



#Writing image to the files in current directory
path = "/home/akshay/Downloads/CV/ps1-sol/"
file = path+image_name+"_grayscale.png"
cv2.imwrite(file, gray_image)
file = path+image_name+"_binary.png"
cv2.imwrite(file, binary_image)
file = path+image_name+"_output.png"
cv2.imwrite(file, output)

#display all the images in 3 screens
cv2.imshow("(b) Grayscale image", gray_image)
cv2.imshow("(c) Black-and-White image", binary_image)
cv2.imshow("(d) Output color image", output)
cv2.waitKey(10000)


