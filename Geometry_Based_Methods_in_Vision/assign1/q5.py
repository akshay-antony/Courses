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
from q3 import find_homography_and_warp

def q5(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--perspective_image_name', type=str,
                        help="image name of the perspective image")
    parser.add_argument('--normal_image_names', type=str,
                        help="a list of normal image names separated by commas")
    args = parser.parse_args(raw_args)
    print(args)
    normal_image_names = args.normal_image_names.split(',')
    perspective_image = cv2.imread(f'./data/q5/{args.perspective_image_name}')
    image_name = args.perspective_image_name.split('.')[0]
    perspective_image_copy = np.copy(perspective_image)
    results_info_filepath = f'./output/q5/results_info/{image_name}.json'
    result_image_filepath = f'./output/q5/results/{image_name}.jpg'
    annotated_image_filepath = f'./output/q5/annotated_images/{image_name}.jpg'

    normal_images = [cv2.imread(f'./data/q5/{normal_image_name}') 
                     for normal_image_name in  normal_image_names]
    results_info = {}
    results_info['perspective_image_name'] = args.perspective_image_name.split('.')[0]
    results_info['normal_image_names'] = args.normal_image_names
    choice = input("Do you want to use the stored annotations (y/n): ")
    if choice == 'n':
        normal_image_points = [[[0, 0],
                                [image.shape[1]-1, 0],
                                [0, image.shape[0]-1],
                                [image.shape[1]-1, image.shape[0]-1]] for image in normal_images]
        
        perspective_image_points = []
        for image_no in range(len(normal_image_names)):
            image_points = []
            cv2.imshow("Normal image", normal_images[image_no])
            print("Select 4 corresponding points on perspective image in order "+ 
                "(top left, top right, bottom left, bottom right)")
            cv2.imshow("Perspective Image", perspective_image_copy)
            cv2.setMouseCallback("Perspective Image", 
                                 select_points_homography, 
                                 (perspective_image_copy, image_points))
            while(1):
                cv2.imshow('Perspective Image', perspective_image_copy)
                k = cv2.waitKey(10) & 0xFF
                if k == 27 or len(image_points)>=4:
                    break
            perspective_image_points.append(image_points)

        results_info['perspective_image_points'] = perspective_image_points
        results_info['normal_image_points'] = normal_image_points
    
    else:
        with open(results_info_filepath, 'r') as f:
            results_info = json.load(f)
        perspective_image_points = results_info['perspective_image_points']
        normal_image_points = results_info['normal_image_points']
    
    annotated_image = draw_annotations_homography(np.copy(perspective_image), 
                                                  np.asarray(perspective_image_points).reshape(-1, 2))
    cv2.imshow("Annotated corners in Perspective Image", annotated_image)
    warped_images = []
    H_s = []
    result = np.copy(perspective_image)
    for image_no in range(len(normal_image_names)):
        warped_image, H = find_homography_and_warp(normal_image_points[image_no],
                                                   perspective_image_points[image_no],
                                                   normal_images[image_no],
                                                   perspective_image)
        warped_images.append(warped_image)
        current_image_points = np.asarray(perspective_image_points[image_no])
        current_image_points[[2, 3], :] = current_image_points[[3, 2], :]
        cv2.fillPoly(result, [current_image_points], (0, 0, 0))
        result = cv2.bitwise_or(result, warped_image)
        H_s.append(H.tolist())
    
    results_info['H'] = H_s
    cv2.imshow("result", result)
    cv2.imwrite(result_image_filepath, result)
    cv2.imwrite(annotated_image_filepath, annotated_image)
    cv2.waitKey(0)
    with open(results_info_filepath, 'w') as f:
        json.dump(results_info, f, indent=3)        

if __name__ == '__main__':
    q5()