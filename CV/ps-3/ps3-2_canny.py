import cv2
import numpy as np
import os
#callback of trackbar
def nothing(x):
    pass

#reading the input image

filename = "/home/akshay/Downloads/CV/ps-3-q/ps3-images/gear.png"
input_image = cv2.imread("/home/akshay/Pictures/i.png")

cv2.namedWindow('image',  cv2.WINDOW_KEEPRATIO)
# create trackbars for color change
cv2.createTrackbar('aperture','image',1,2,nothing)
#0-False,1-True for L2norm
cv2.createTrackbar('L2norm','image',0,1,nothing)
cv2.createTrackbar('threshold1','image',0,255,nothing)
cv2.createTrackbar('threshold2','image',0,255,nothing)
#aperture size limited in (3,5,7) in Canny function
#so the result from trackbar aperture is modified as 
#2*aperture+3

threshold1,threshold,L2,aperture = 0,0,0,0
#Infinite loop activated whenever a trackbar value is changed
while(1):
    # get current positions of four trackbars
    threshold1 = cv2.getTrackbarPos('threshold1','image')
    threshold2 = cv2.getTrackbarPos('threshold2','image')
    aperture = cv2.getTrackbarPos('aperture','image')
    L2 = bool(cv2.getTrackbarPos('L2norm','image'))
    #applying cv2 canny 
    new_image = cv2.Canny(input_image,threshold1,threshold2,apertureSize=2*aperture+3,L2gradient=L2)
    # flipping
    #new_image = np.uint8(np.where(new_image>127,0,255))
    cv2.imshow("image",new_image)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
#saving the image
first_name = os.path.basename(filename)
first_name = first_name.split('.',1)[0]
write_filename = first_name + "-canny.png"
print(threshold1,threshold2,aperture,L2)
if(cv2.imwrite(write_filename,new_image)):
    print("Successfully saved")
else:
    print("Save Unsuccessfull")
