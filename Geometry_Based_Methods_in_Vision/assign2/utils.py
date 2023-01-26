from turtle import color
import numpy as np
import random
import cv2


def to_homogenous(input_array):
    # expects numpy array
    if isinstance(input_array, list):
        input_array = np.asarray(input_array) # N*2
    return np.concatenate((input_array, np.ones((input_array.shape[0], 1))), axis=1)

def standardize(data, mean_req=(0, 0), std_req=(np.sqrt(2), np.sqrt(2))):
    if isinstance(data, list):
        data = np.asarray(data) # N*2
    mean_initial = np.mean(data, axis=0) 
    std_initial = np.std(data, axis=0)
    ## transformation matrix
    H = np.asarray(
            [[std_req[0]/std_initial[0], 0, mean_req[0]-(std_req[0]/std_initial[0])*mean_initial[0]],
            [0, std_req[1]/std_initial[1], mean_req[1]-(std_req[1]/std_initial[1])*mean_initial[1]],
            [0, 0, 1]]
        )
    return H


def batch_transformation(H, points):
    # funtion to transform a batch of points
    if len(points.shape) == 2:
        points = points.reshape(-1, points.shape[1], 1)
        transformed_points = np.einsum('MN, BNi -> BMi', H, points)
    return transformed_points.reshape(-1, points.shape[1])

def angle_between_planes(plane1, plane2):
    cos_angle = np.dot(plane1, plane2) / \
                (np.linalg.norm(plane1)*np.linalg.norm(plane2))
    return np.rad2deg(np.arccos(cos_angle))

def plot_planes(plane, img, plane_no):
    if isinstance(plane, list):
        plane = np.asarray(plane)

    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    rand_color = (r, g, b)
    for i in range(4):
        cv2.line(img, (plane[i, 0], plane[i, 1]),
                 (plane[(i+1)%4, 0], plane[(i+1)%4, 1]),
                 rand_color, thickness=5)
    cv2.putText(img, str(plane_no+1), 
                (int(np.mean(plane[:, 0])), int(np.mean(plane[:, 1]))), 
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 5, cv2.LINE_AA)
    return img

if __name__ == '__main__':
    pass