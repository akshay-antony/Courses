import cv2  
import numpy as np
import json 

def normalize(points):
    '''
    points of size n*2
    '''
    x0, y0 = np.mean(points[:, 0]), np.mean(points[:, 1])
    d_avg = np.mean(np.sqrt((points[:, 0] - x0)**2 +
                            (points[:, 1] - y0)**2))
    s = np.sqrt(2) / d_avg
    T = np.asarray([[s, 0, -s*x0],
                    [0, s, -s*y0],
                    [0, 0, 1]])
    return T

def to_homogenous(x):
    return np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)


def plot_epipolar(image_1, 
                  image_2,
                  annotations_filename,
                  F):
    image_annotated = np.copy(image_1)
    annotated_line = np.copy(image_2)
    choice = input("do you want to use saved points (y/n): ")
    cv2.imshow("image_annotated", image_annotated)
    point_list = []
    if choice == 'n':
        num_points = input("Enter number of annotations you need ")
        while len(point_list) < int(num_points):
            print(f"Select {len(point_list)+1} point on image")
            color = (0, 0, 255)
            cv2.setMouseCallback('image_annotated', select_points, (image_annotated, point_list, color))
            color = np.random.randint(0, 255, (3,)).tolist()

            while(1):
                # cv2.imshow('image 1', image_1)
                cv2.imshow("image_annotated", image_annotated)
                cv2.imshow("line_annotated",  annotated_line)
                k = cv2.waitKey(10) & 0xFF
                if k == 27 or len(point_list) == int(num_points):
                    break
                if len(point_list) >= 1:
                    point = point_list[-1]
                    image_annotated = cv2.circle(image_annotated,
                                                (point[0], point[1]),
                                                radius=10,
                                                color=color, 
                                                thickness=-1)
                    image_annotated = cv2.putText(image_annotated, 
                                                str(len(point_list)),
                                                (point[0], point[1]),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                color,
                                                3)
                    line_endpoints = calculate_line(point, image_2, F)
                    annotated_line = cv2.line(annotated_line, 
                                            (line_endpoints[0][0], line_endpoints[0][1]),
                                            (line_endpoints[1][0], line_endpoints[1][1]),
                                            color=color,
                                            thickness=2)
                    annotated_line = cv2.putText(annotated_line, 
                                                str(len(point_list)),
                                                ((line_endpoints[0][0]+line_endpoints[1][0])//2,
                                                (line_endpoints[0][1]+line_endpoints[1][1])//2),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                color,
                                                3)
            


        with open(annotations_filename, 'w') as f:
                json.dump(point_list, f, indent=4)

    else:
        with open(annotations_filename, 'r') as f:
            point_list = json.load(f)

    image_annotated = np.copy(image_1)
    annotated_line = np.copy(image_2)
    for point_no in range(len(point_list)):
        color = np.random.randint(0, 255, (3,)).tolist()
        image_annotated = cv2.circle(image_annotated,
                                    (point_list[point_no]),
                                    radius=10,
                                    color=color, 
                                    thickness=-1)
        image_annotated = cv2.putText(image_annotated, 
                                      str(point_no+1),
                                      (point_list[point_no]),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      1,
                                      color,
                                      3,
                                    )
        line_endpoints = calculate_line(point_list[point_no], 
                                        image_2, F)
        annotated_line = cv2.line(annotated_line, 
                                (line_endpoints[0][0], line_endpoints[0][1]),
                                (line_endpoints[1][0], line_endpoints[1][1]),
                                color=color,
                                thickness=2)
        annotated_line = cv2.putText(annotated_line, 
                                    str(point_no+1),
                                    ((line_endpoints[0][0]+line_endpoints[1][0])//2,
                                    (line_endpoints[0][1]+line_endpoints[1][1])//2),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    color,
                                    3,
                                    )
    # cv2.waitKey(0)
    # cv2.destroyWindow("image_annotated")
    # cv2.destroyWindow("line_annotated")
    return image_annotated, annotated_line

def calculate_line(point, 
                   image_2, 
                   F):
    point = np.asarray([point[0], point[1], 1])
    line_2 = F @ point
    line_2 /= line_2[-1] 
    line_endpoints = [[0, 0], [0, 0]]
    # first point
    line_endpoints[0][1] = int(-line_2[2] / line_2[1])
    line_endpoints[1][0] = image_2.shape[1]-1
    line_endpoints[1][1] = int(-(line_2[2]+line_endpoints[1][0]*line_2[0]) / line_2[1])
    return line_endpoints

def select_points(event, 
                  x, 
                  y, 
                  flags, 
                  params):
    image = params[0]
    # list to append point to
    line_points = params[1]
    if event == cv2.EVENT_LBUTTONUP:
        print(x, y)
        if len(params) == 3:
            cv2.circle(image, (x, y), 0, params[2], 10)
        else:
            cv2.circle(image, (x, y), 0, (0, 255, 0), 10)
        line_points.append([x, y])

def print_np_array(array, prec=4):
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            print(round(array[row, col], prec))

if __name__ == '__main__':
    pass