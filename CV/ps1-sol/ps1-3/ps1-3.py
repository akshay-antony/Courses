import cv2
import numpy as np
import os


#Taking the filename and gamma from the user
file_name = input("Please input the location of the image: ")
#gamma for the dark image is around 2.0 and for the carnival image is around .5
gamma = input("Please input the gamma value: ")

#extracting the image name
image_name = os.path.basename(file_name)
image_name = image_name.split('.',1)[0]

#Reading the input image
image_input = cv2.imread(file_name)

#Making the image between 0-1
n_image = image_input/255.0
#Applying gamma correction by raising to the power of
n_image = np.power(n_image,1/float(gamma))
#Converting back to 0-255
n_image *= 255
#Making the data type 8-bit
n_image = np.asarray(n_image, dtype="uint8")

#Location of the current directory
path = "/home/akshay/Downloads/CV/ps1-sol/"
file = path+image_name+"_gcorrected.jpg"
#Writing the image
cv2.imwrite(file,n_image)

#Showing the orginal input and gamma corrected
cv2.imshow("(a)Input image",image_input)
cv2.imshow("(b) Output image after gamma correction",n_image)
cv2.waitKey(0)


