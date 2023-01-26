import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import to_homogenous
from annotations import vis_annotations_q2a, vis_annotations_given_points
import json


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

def make_constraints(u, v, A):
    constraint = [u[0]*v[0] + u[1]*v[1], 
                  u[0]*v[2] + u[2]*v[0], 
                  u[1]*v[2] + u[2]*v[1], 
                  u[2]*v[2]]
    constraint = np.asarray(constraint).reshape(-1, 4)
    A = np.concatenate((A, constraint), axis=0)
    return A

def find_vanishing_point(points):
    '''
    points = 4*2 numpy array
    '''
    if isinstance(points, list):
        points = np.asarray(points)
    lines = []
    for i in range(2):
        point1 = to_homogenous(points[2*i].reshape(1, -1))
        point2 = to_homogenous(points[2*i+1].reshape(1, -1))
        lines.append(np.cross(point1, point2))
    
    lines = np.asarray(lines)
    vanishing_point = np.cross(lines[0], lines[1])
    return vanishing_point

def main():
    image_name = "q2a"
    image_filename = f"./data/{image_name}.png"
    annotations_filename = f"output/q2/annotations/annotations{image_name}.json"
    results_filename = f'./output/results/{image_name}.png'
    input_image = cv2.imread(image_filename)
    image_copy = np.copy(input_image)
    choice = input("do you want to use saved points (y/n): ")

    cv2.imshow("input_image", image_copy)
    line_points = []
    if choice == 'n':
        for i in range(3):
            print("Select 4 points for set parallel lines")
            # selecting points for first two parallel lines
            if i == 0:
                color = (0, 0, 255)
            elif i == 1:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            
            cv2.setMouseCallback('input_image', select_points, (image_copy, line_points, color))
            while(1):
                cv2.imshow('input_image', image_copy)
                k = cv2.waitKey(10) & 0xFF
                if k == 27 or len(line_points) == (i+1)*4:
                    break
    
        with open(annotations_filename, 'w') as f:
                json.dump(line_points, f, indent=4)

    else:
        with open(annotations_filename, 'r') as f:
            line_points = json.load(f)
    line_points = [[417, 514],  
                   [602, 410],
                   [637, 687], 
                   [1011, 537],
                   [574, 398],
                   [572, 209],
                   [735, 481],  
                   [713, 303],
                   [315, 457], 
                   [44, 293],
                   [38, 424],
                   [326, 577]]
    vis_annotations_given_points(line_points, input_image)
    vanishing_points = []
    for i in range(len(line_points)//4):
        vanishing_point = find_vanishing_point(line_points[4*i: 4*i+4])
        vanishing_points.append(vanishing_point)
    vanishing_points = np.asarray(vanishing_points).reshape(-1, 3)
    vanishing_points /= vanishing_points[:, 2].reshape(-1, 1)
    print(vanishing_points)
    x_max, x_min = int(np.max(vanishing_points[:, 0])), int(np.min(vanishing_points[:, 0]))
    y_max, y_min = int(np.max(vanishing_points[:, 1])), int(np.min(vanishing_points[:, 1]))
    image_result = np.ones((y_max-y_min+100, x_max-x_min+100, 3))*255
    # image_result[np.int16(vanishing_points[0, 0]), :] = [0, 0, 255]
    image_result = np.uint8(image_result)
    print(x_max-x_min, y_max-y_min, image_result.shape)

    for i in range(3):
        print(i)
        cv2.circle(image_result, 
                   (int(vanishing_points[i, 0]-x_min+50), 
                    int(vanishing_points[i, 1]-y_min+50)), 
                    20, (0, 0, 255), -1)
        cv2.line(image_result, 
                 (int(vanishing_points[i, 0]-x_min+50), 
                  int(vanishing_points[i, 1]-y_min+50)), 
                 (int(vanishing_points[(i+1)%3, 0]-x_min+50), 
                  int(vanishing_points[(i+1)%3, 1]-y_min+50)), 
                 (255, 0, 0), thickness=5)
        print(int(vanishing_points[i, 0]-x_min+50), int(vanishing_points[i, 1]-y_min+50))
    # image_result = cv2.resize(image_result, (400, 800))
    image_result[-x_min+50:-x_min+input_image.shape[0]+50, -y_min+50:-y_min+input_image.shape[1]+50, :] = input_image
    principle_point = [int(np.mean(vanishing_points[:, 1]) - y_min+50), 
                       int(np.mean(vanishing_points[:, 0]) - x_min+50)]
    cv2.circle(image_result, principle_point, 20, (0, 255, 0), -1)

    A = np.zeros((0, 4))
    A = make_constraints(vanishing_points[0], vanishing_points[1], A)
    A = make_constraints(vanishing_points[0], vanishing_points[2], A)
    A = make_constraints(vanishing_points[1], vanishing_points[2], A)
    print(A.shape)
    _, _, v = np.linalg.svd(A)
    v = v[-1, :]
    omega = np.asarray([[v[0], 0, v[1]],
                        [0, v[0], v[2]],
                        [v[1], v[2], v[3]]])
    omega /= omega[-1, -1]
    L = np.linalg.cholesky(omega)
    K = np.linalg.inv(L.T)
    K /= K[-1, -1]
    for row in range(3):
        for col in range(3):
            print(K[row, col])
    cv2.imshow("final", image_result)
    cv2.waitKey(0)
    cv2.imwrite("./output/q2/results/vanishing_and_principal.jpg", image_result)
    # for i in range(vanishing_points.shape[0]):
    #     plt.plot(vanishing_points[i, 0], vanishing_points[i, 1], marker='o')
    # plt.imshow(input_image)
    # plt.show()

if __name__ == '__main__':
    main()