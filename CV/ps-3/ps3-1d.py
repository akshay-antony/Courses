import cv2
import numpy as np
import os


if __name__ == '__main__':
	original_image = cv2.imread("/home/akshay/Downloads/CV/ps-3-q/ps3-images/rainbow.png")
	cv2.imshow("input image",original_image)

	#blurred_image = cv2.medianBlur(original_image,3)
	blurred_image = cv2.bilateralFilter(original_image,9,75,75)
	#blurred_image = cv2.GaussianBlur(original_image,ksize=(3,3),sigmaX=0)
	#blurred_image = cv2.boxFilter(original_image,-1,(3,3))
	cv2.imshow("blurred_image",blurred_image)
	sharpen_matrix = np.asarray([[0,-1,0],[-1,5,-1],[0,-1,0]])
	improved_image = cv2.filter2D(blurred_image,-1,sharpen_matrix)
	cv2.imshow("improved image",improved_image)
	cv2.waitKey(0)