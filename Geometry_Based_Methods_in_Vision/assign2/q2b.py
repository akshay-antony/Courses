import numpy as np
import cv2
import json
from utils import standardize, \
                  to_homogenous,\
                  batch_transformation, \
                  angle_between_planes
from annotations import vis_annnotations_q2b_given_points

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

def find_orientation(square_points, K):
    '''
    square points = top left -> top right ->
                    bottom right -> bottom left
    '''
    if isinstance(square_points, list):
        square_points = np.asarray(square_points)

    square_points = to_homogenous(square_points)
    line1 = np.cross(square_points[0], square_points[1])
    line2 = np.cross(square_points[2], square_points[3])
    vanishing_point_1 = np.cross(line1, line2)

    line1 = np.cross(square_points[1], square_points[2])
    line2 = np.cross(square_points[0], square_points[3])
    vanishing_point_2 = np.cross(line1, line2)
    d1 = np.linalg.inv(K) @ vanishing_point_1
    d2 = np.linalg.inv(K) @ vanishing_point_2
    n = np.cross(d1, d2)
    return n

def make_constraints(A, homography):
    constraint_1 = [homography[0, 0] * homography[0, 1], 
                    homography[1, 1] * homography[0, 0] + homography[1, 0] * homography[0, 1],
                    homography[0, 0] * homography[2, 1] + homography[2, 0] * homography[0, 1], 
                    homography[1, 0] * homography[1, 1],
                    homography[1, 0] * homography[2, 1] + homography[2, 0] * homography[1, 1],
                    homography[2, 0] * homography[2, 1]]
    constraint_2 = [homography[0, 0]**2 - homography[0, 1]**2,
                    2*homography[0, 0]*homography[1, 0] - 2*homography[0, 1]*homography[1, 1],
                    2*homography[0, 0]*homography[2, 0] - 2*homography[0, 1]*homography[2, 1],
                    homography[1, 0]**2 - homography[1, 1]**2,
                    2*homography[1, 0]*homography[2, 0] - 2*homography[1, 1]*homography[2, 1],
                    homography[2, 0]**2 - homography[2, 1]**2]
    constraint_2 = np.asarray(constraint_2).reshape(-1, 6)
    constraint_1 = np.asarray(constraint_1).reshape(-1, 6)
    A = np.concatenate((A, constraint_1, constraint_2), axis=0)
    return A

def find_homography(first_image_points,
                    second_image_points):
    T1 = standardize(first_image_points)
    T2 = standardize(second_image_points)
    first_points_homogenous = to_homogenous(first_image_points)
    second_points_homogenous = to_homogenous(second_image_points)
    first_points_homogenous =  batch_transformation(T1, first_points_homogenous)
    second_points_homogenous = batch_transformation(T2, second_points_homogenous)
    # construction x' * x h = 0 for svd
    A = np.zeros((first_points_homogenous.shape[0]*2, 9))
    for i in range(first_points_homogenous.shape[0]):
        A[2*i, 3:6] = -second_points_homogenous[i, 2] * first_points_homogenous[i] 
        A[2*i, 6:9] = second_points_homogenous[i, 1] * first_points_homogenous[i]
        A[2*i+1, 0:3] = second_points_homogenous[i, 2] * first_points_homogenous[i]
        A[2*i+1, 6:9] = -second_points_homogenous[i, 0] * first_points_homogenous[i]
    #print(A)

    _, _, vh = np.linalg.svd(A)
    H = vh[-1, :].reshape(3, 3)
    full_transformation = np.linalg.inv(T2) @ H @ T1
    full_transformation /= full_transformation[-1, -1]
    return full_transformation

def main():
    image_name = "q2b"
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
            print("Select 4 points for a square")
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

    sqaure_points = [[0, 1], [1, 1], [1, 0], [0, 0]]
    homographies = []
    A = np.zeros((0, 6))
    for i in range(3):
        homography = find_homography(sqaure_points,
                                     line_points[4*i: 4*i+4])
        A = make_constraints(A, homography)
        homographies.append(homography)
    # print(homographies, A.shape)
    # A = A[:-1, :]
    print(A.shape)
    _, _, v = np.linalg.svd(A)
    v = v[-1, :]
    omega = np.asarray([[v[0], v[1], v[2]],
                        [v[1], v[3], v[4]],
                        [v[2], v[4], v[5]]])
    omega /= v[5]
    L = np.linalg.cholesky(omega)
    K = np.linalg.inv(L.T)
    K /= K[-1, -1]
    for row in range(3):
        for col in range(3):
            print(K[row, col])
    directions = []
    for i in range(3):
        vis_annnotations_q2b_given_points(line_points[4*i: 4*i+4], 
                                          np.copy(input_image), i)
        direction = find_orientation(line_points[4*i: 4*i+4], K)
        directions.append(direction)
    directions = np.asarray(directions)
    directions /= directions[:, 2].reshape((-1, 1))

    plane1_2 = angle_between_planes(directions[0], directions[1])
    plane2_3 = angle_between_planes(directions[1], directions[2])
    plane1_3 = angle_between_planes(directions[0], directions[2])
    print(plane1_2, plane2_3, plane1_3)

if __name__ == '__main__':
    main()