import numpy as np
import cv2
import os 
import argparse
from utils import to_homogenous, standardize
import matplotlib.pyplot as plt


def project(camera_matrix, points, image):
    '''
    camera_matrix=3*4
    points = N*4
    image = H*W*3
    '''
    image_coords = (camera_matrix @ points.T).T
    image_coords /= image_coords[:, -1].reshape((-1, 1))
    image_coords = np.int32(image_coords)
    for coord in image_coords:
        image = cv2.circle(image, coord[:2], 2, (0, 0, 255), -1)
    return image

def project_bbox(camera_matrix, points, image):
    '''
    points = N*6
    '''
    for point in points:
        point_project_1 = (camera_matrix @ to_homogenous(point[:3].reshape(-1, 3)).T)
        point_project_1 /= point_project_1[-1]
        point_project_2 = (camera_matrix @ to_homogenous(point[3:].reshape(-1, 3)).T)
        point_project_2 /= point_project_2[-1]
        image = cv2.line(image, (int(point_project_1[0]), 
                                 int(point_project_1[1])), 
                                (int(point_project_2[0]), 
                                 int(point_project_2[1])),
                                 (255, 0, 0), 2)
    return image 

def main():
    image_name = "cube"
    image_filename = f"./data/q1/{image_name}.jpeg"
    correspondence_filename = f"./data/q1/{image_name}.txt"
    test_points_filename = f"./data/q1/{image_name}_pts.npy"
    test_bbox_points_filename = "./data/q1/bunny_bd.npy"
    correspondences = np.loadtxt(correspondence_filename)
    image_coords = correspondences[:, :2]
    point_coords = correspondences[:, 2:]
    image_coords = to_homogenous(image_coords)
    # image_coords = H @ image_coords.T
    point_coords = to_homogenous(point_coords)

    A = np.zeros((2*point_coords.shape[0], 12))
    for i in range(image_coords.shape[0]):
        A[2*i, 4:8] = -image_coords[i, 2] * point_coords[i]
        A[2*i, 8:12] = image_coords[i, 1] * point_coords[i]
        A[2*i+1, 0:4] = image_coords[i, 2] * point_coords[i]
        A[2*i+1, 8:12] = -image_coords[i, 0] * point_coords[i]

    _, _, v = np.linalg.svd(A)
    camera_matrix = v[-1, :]
    camera_matrix = np.reshape(camera_matrix, (3, 4))
    camera_matrix /= camera_matrix[-1, -1]
    print(camera_matrix)
    for k in range(camera_matrix.shape[0]):
        for j in range(camera_matrix.shape[1]):
            print(camera_matrix[k][j])
    ###
    inference_result = (camera_matrix@point_coords.T).T
    inference_result /= inference_result[:, -1].reshape((-1, 1))
    inference_result = inference_result.astype(np.int16)
    ###

    test_points = np.load(test_points_filename)
    test_bbox_points = np.load(test_bbox_points_filename)
    test_points = to_homogenous(test_points)
    test_image = cv2.imread(image_filename)
   
    # test_image[inference_result[:, 2], inference_result[:, 1], :] = [0, 0, 255]
    cv2.imshow("a", test_image)
    test_image_copy = np.copy(test_image)
    image_projected = project(camera_matrix, test_points, test_image)
    image_bbox_projected = project_bbox(camera_matrix, test_bbox_points, test_image_copy)
    cv2.imshow("Surface Points", image_projected)
    cv2.waitKey(0)
    cv2.imwrite(f"{image_name}_bbox_project.jpg", image_bbox_projected)
    cv2.imwrite(f"{image_name}_project.jpg", image_projected)

if __name__ == '__main__':
    # cube_image = cv2.imread("./data/q1/cube_custom.jpg")
    # plt.imshow(cube_image)
    # plt.show()
    # cv2.imshow("cube", cube_image)
    # cv2.waitKey(0)
    main()