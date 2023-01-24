import cv2
import numpy as np
import os


def main():
	filename = "/home/akshay/Pictures/i.png"
	original_image = cv2.imread(filename)
	cv2.imshow("original_image",original_image)
	#after trying found that medianBlur is the best
	blurred_image = cv2.medianBlur(original_image,3)
	#defining the sharpening matrix
	sharpen_matrix = np.asarray([[-1.,-1.,-1.],[-1.,9.,-1.],[-1.,-1.,-1.]])
	smooth_sharped_image = cv2.filter2D(blurred_image,-1,sharpen_matrix)

	cv2.imshow("smooth and sharped image",smooth_sharped_image)

	first_name = os.path.basename(filename)
	first_name = first_name.split('.',1)[0]
	write_filename = first_name + "-improved.png"
	
	if(cv2.imwrite(write_filename,smooth_sharped_image)):
		print("Successfully saved")
	else:
		print("Save Unsuccessfull")

	cv2.waitKey(0)

if __name__ == '__main__':
	main()