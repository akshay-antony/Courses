import numpy as np
import cv2

COLORS = [(0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 0), (0, 0, 255)]
def vis_annotations_q2a():
	'''
	annotations: (3, 2, 4) 
	3 groups of parallel lines, each group has 2 lines, each line annotated as (x1, y1, x2, y2)
	'''
	annotations = np.load('data/q2/q2a.npy')			
	img = cv2.imread('data/q2a.png')
	for i in range(annotations.shape[0]):
		COLOR = COLORS[i]
		lines = annotations[i]
		for j in range(lines.shape[0]):
			x1, y1, x2, y2 = lines[j]
			cv2.circle(img, (x1, y1), 3, COLOR, -1)
			cv2.circle(img, (x2, y2), 3, COLOR, -1)
			cv2.line(img, (x1, y1), (x2, y2), COLOR, 2)

	cv2.imshow('q2a', img)
	cv2.waitKey(0)


def vis_annotations_given_points(line_points, img):
	line_points = np.asarray(line_points)
	for i in range(line_points.shape[0]//2):
		COLOR = COLORS[i//2]
		x1 = line_points[2*i][0]
		y1 = line_points[2*i][1]
		x2 = line_points[2*i+1][0]
		y2 = line_points[2*i+1][1]
		cv2.circle(img, (x1, y1), 3, COLOR, -1)
		cv2.circle(img, (x2, y2), 3, COLOR, -1)
		cv2.line(img, (x1, y1), (x2, y2), COLOR, 2)
	cv2.imwrite("./output/q2/annotations/q2.jpg", img)
	# while 1:
	# 	cv2.imshow('q2a', img)
	# 	k = cv2.waitKey(10) & 0xFF
	# 	if k == 27:
	# 		break

def vis_annnotations_q2b():
	'''
	annotations: (3, 4, 2) 
	3 squares, 4 points for each square, each point annotated as (x, y).
		pt0 - pt1
		|		|
		pt3 - pt2
	'''
	annotations = np.load('data/q2/q2b.npy').astype(np.int64)		
	img = cv2.imread('data/q2b.png')

	for i in range(annotations.shape[0]):
		COLOR = COLORS[i]
		square = annotations[i]
		for j in range(square.shape[0]):
			x, y = square[j]		 
			cv2.circle(img, (x, y), 3, COLOR, -1)
			cv2.putText(img, str(j), (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, COLOR, 1, cv2.LINE_AA)
			cv2.line(img, square[j], square[(j+1) % 4], COLOR, 2)

	cv2.imshow('q2b', img)
	cv2.waitKey(0)

def vis_annnotations_q2b_given_points(annotations, img, img_no=0):
	if isinstance(annotations, list):
		annotations = np.asarray(annotations)
	COLOR = COLORS[0]
	points = np.int32(annotations).reshape(-1, 1, 2)
	for i in range(annotations.shape[0]): 
		x, y = annotations[i, :]	 
		cv2.circle(img, (x, y), 3, COLOR, -1)
		cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, COLOR, 1, cv2.LINE_AA)
		cv2.line(img, annotations[i], annotations[(i+1) % 4], COLOR, 2)
	cv2.fillPoly(img, points, (255, 255, 255))
	cv2.imshow('q2b', img)
	cv2.imwrite(f"./output/q2/annotations/squares{img_no}.jpg", img)
	cv2.waitKey(0)

def vis_annotations_q3():
	'''
	annotations: (5, 4, 2)
	5 planes in the scene, 4 points for each plane, each point annotated as (x, y).
		pt0 - pt1
		|		|
		pt3 - pt2
	'''
	annotations = np.load('data/q3/q3.npy').astype(np.int64)		
	img = cv2.imread('data/q3.png')

	for i in range(annotations.shape[0]):
		COLOR = COLORS[i]
		square = annotations[i]
		for j in range(square.shape[0]):
			x, y = square[j]		 
			cv2.circle(img, (x, y), 3, COLOR, -1)
			cv2.putText(img, str(j+i*4), (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, COLOR, 1, cv2.LINE_AA)
			cv2.line(img, square[j], square[(j+1) % 4], COLOR, 2)

		cv2.imshow('q3', img)
		cv2.waitKey(0)

if __name__ == '__main__':
	vis_annotations_q2a()
	vis_annnotations_q2b()
	vis_annotations_q3()