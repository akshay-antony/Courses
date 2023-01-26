import numpy as np
import json
import cv2
from utils import MyWarp, select_points_homography, \
                  draw_annotations_homography, \
                  standardize, \
                  to_homogenous, \
                  batch_transformation, \
                  warp_for_homography, \
                  make_points_cyclic
import argparse

def find_homography_and_warp(first_image_points,
                             second_image_points,
                             first_image,
                             second_image):
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
    output = warp_for_homography(first_image, full_transformation, (second_image.shape[:2]))
    return output, full_transformation

def q3(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--first_image_name', type=str,
                        help='name of the normal image')
    parser.add_argument('--second_image_name', type=str,
                        help='name of the perspective image')
    print(raw_args)
    args = parser.parse_args(raw_args)

    first_image_filename = f"./data/q3/{args.first_image_name}"
    second_image_filename = f"./data/q3/{args.second_image_name}" 
    image_base_name = args.first_image_name.split('.')[0]

    result_info_filepath = f'./output/q3/results_info/{image_base_name}.json'  
    result_filepath = f'./output/q3/warped_results/{image_base_name}.jpg'
    annotated_image_filepath = f'./output/q3/input_image_annotated/{image_base_name}.jpg'
    first_image = cv2.imread(first_image_filename)
    second_image = cv2.imread(second_image_filename)
    second_image_copy = np.copy(second_image)
    choice = input("Do you want to use saved annotations (y/n): ")
    results_info = {}

    if choice == 'n':
        first_image_points = []
        first_image_points.append([0, 0])
        first_image_points.append([first_image.shape[1]-1, 0])
        first_image_points.append([0, first_image.shape[0]-1])
        first_image_points.append([first_image.shape[1]-1, first_image.shape[0]-1])

        second_image_points = []
        cv2.imshow("Normal Image", first_image)
        cv2.imshow("Perspective Image", second_image_copy)
        print("Please select points on second image in following order " +  
              "(top left, top right, bottom left, bottom right)")
        cv2.setMouseCallback("Perspective Image", 
                             select_points_homography, 
                             (second_image_copy, second_image_points))
        while(1):
            cv2.imshow('Perspective Image', second_image_copy)
            k = cv2.waitKey(10) & 0xFF
            if k == 27 or len(second_image_points)>=4:
                break
        assert(len(second_image_points) == 4)
        results_info['first_image_points'] = first_image_points
        results_info['second_image_points'] = second_image_points
    
    else:
        with open(result_info_filepath, 'r') as f:
            results_info = json.load(f)
        first_image_points = results_info['first_image_points']
        second_image_points = results_info['second_image_points']

    annotated_second_image = draw_annotations_homography(second_image_copy, second_image_points)
    while(1):
        cv2.imshow('Annotated corners in Perspective Image', annotated_second_image)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break

    output, full_transformation = find_homography_and_warp(first_image_points,
                                                           second_image_points,
                                                           first_image,
                                                           second_image)
    result = np.copy(second_image)
    second_image_points = np.asarray(second_image_points)
    cyclic_indices = make_points_cyclic(second_image_points)
    second_image_points = second_image_points[cyclic_indices].reshape(-1, 1, 2)
    cv2.fillPoly(result, [second_image_points], (0, 0, 0))
    result = cv2.bitwise_or(result, output)
    results_info['H'] = full_transformation.tolist()
    with open(result_info_filepath, 'w') as f:
        json.dump(results_info, f, indent=3)
    cv2.imshow("Warped and Overlaid Image", result)
    cv2.imwrite(result_filepath, result)
    cv2.imwrite(annotated_image_filepath, annotated_second_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    q3()