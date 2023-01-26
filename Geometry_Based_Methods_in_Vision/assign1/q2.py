import numpy as np
import cv2 
import json
import argparse
from utils import MyWarp, \
                  normalize,\
                  normalize_projective,\
                  select_points,\
                  find_cosine_before_and_after,\
                  draw_annotations, \
                  annotated_rectified_image
from affine_rectification import find_affine_rectification_matrix


def q2(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', default='tile5',
                        help="name of the image to recitify")

    args = parser.parse_args(raw_args)

    # filenames to read from
    image_filename = f"./data/q1/{args.image_name}.jpg"
    # parall annotations
    affine_annotations_filename = f'./data/q1_annotations/{args.image_name}.json'
    
    # filenames to write to
    # perpendicular annotations in original image
    annotations_filename = f'./data/q2_input_annotations/{args.image_name}.json'
    annotated_input_image = f'./output/q2/input_image_annotated/{args.image_name}.jpg'
    annotated_intermediate_filename = f'./output/q2/affined_rectified_annotated/{args.image_name}.jpg'
    output_file_path = f'./output/q2/similarity_rectified/{args.image_name}.jpg'
    results_info_filename = f'./output/q2/results_info/{args.image_name}.json'
    cosine_annotated_filename = f'./output/q2/cosine_marked/{args.image_name}.jpg'
    cosine_on_input_filename = f'./output/q2/cosines_on_input/{args.image_name}.jpg'

    input_image = cv2.imread(image_filename)
    image_copy_2 = np.copy(input_image)
    print("Using affine rectified image")
    input_image_copy = np.copy(input_image)
    with open(affine_annotations_filename, 'r') as f:
        line_points_parallel = json.load(f)

    affine_rectfied_transformation = find_affine_rectification_matrix(line_points_parallel)
    image_affine_rectified, affine_rectfied_transformation = MyWarp(input_image, 
                                                                    affine_rectfied_transformation,
                                                                    True)
    choice = input("Do you want to use stored annotations (y/n): ")
    line_points = []

    if choice == 'n':
        cv2.imshow("input_image", input_image_copy)
        print("Please select first pair of perpendicular line")
        cv2.setMouseCallback("input_image", select_points, (input_image_copy, line_points))
        while(1):
            cv2.imshow('input_image', input_image_copy)
            k = cv2.waitKey(10) & 0xFF
            if k == 27 or len(line_points)>=4:
                break
        assert(len(line_points) == 4)

        print("Please select second pair of perpendicular line")
        cv2.setMouseCallback("input_image", select_points, (input_image_copy, line_points, 'r'))
        while(1):
            cv2.imshow('input_image', input_image_copy)
            k = cv2.waitKey(10) & 0xFF
            if k == 27 or len(line_points)>=8:
                break
        assert(len(line_points) == 8)

        with open(annotations_filename, 'w') as f:
            json.dump(line_points, f, indent=4)
    
    else:
        with open(annotations_filename, 'r') as f:
            line_points = json.load(f)
            assert(len(line_points) == 8) 

    line_points_homogenous = np.concatenate([np.asarray(line_points), 
                                                np.ones((len(line_points), 1))], axis=1).reshape(-1, 3, 1)
    line_points_transformed = np.einsum('MN, BNi -> BMi', 
                                            affine_rectfied_transformation, 
                                            line_points_homogenous).reshape(-1, 3)
    line_points_transformed = line_points_transformed / line_points_transformed[:, 2].reshape(-1, 1)
    line_points_transformed = line_points_transformed[:, :2].tolist()
    
    annotated_image = draw_annotations(input_image_copy, line_points)
    while 1:
        cv2.imshow("Annotated perpendicular lines on input image", annotated_image)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break
    
    annotations_on_affine_rectified_image = draw_annotations(np.copy(image_affine_rectified), line_points_transformed)
    while 1:
        cv2.imshow("Annotated perpendicular lines on Affine-Rectified Image", 
                    annotations_on_affine_rectified_image)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break

    # finding perpendicular lines
    line_equations = []
    for i in range(len(line_points_transformed)//2):
        line_equation = np.cross(np.asarray([line_points_transformed[2*i][0],
                                                line_points_transformed[2*i][1],
                                                1]),
                                    np.asarray([line_points_transformed[2*i+1][0], 
                                                line_points_transformed[2*i+1][1],
                                                1]))
        line_equations.append(normalize(line_equation))
    
    assert(len(line_equations) == 4)
    A = np.zeros((len(line_equations)//2, 3))
    
    for i in range(len(line_equations) // 2):
        # A[i] = l[i][0]*m[i][0], l[i][1]*m[i][0] + l[i][0]*m[i][1], l[i][1]*m[i][1]
        A[i, 0] = line_equations[2*i][0] * line_equations[2*i+1][0]
        A[i, 1] = line_equations[2*i][1] * line_equations[2*i+1][0] + line_equations[2*i][0] * line_equations[2*i+1][1]
        A[i, 2] = line_equations[2*i][1] * line_equations[2*i+1][1]
    
    _, singular_values, vh = np.linalg.svd(A)
    req_eigen_vector = vh[-1, :] / vh[-1, -1]
    C_infinity = np.asarray([[req_eigen_vector[0], req_eigen_vector[1], 0],
                             [req_eigen_vector[1], req_eigen_vector[2], 0],
                             [0,                   0,                   0]])
    # print(singular_values, C_infinity)
    u, s, uh = np.linalg.svd(C_infinity)
    # print(u, s, uh)
    diagonal_singular = np.asarray([[1/np.sqrt(s[0]), 0, 0],
                                    [0, 1/np.sqrt(s[1]), 0],
                                    [0, 0, 1]]) 
    H = diagonal_singular @ uh
    conic_infinity = np.asarray([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 0]])
    C_infinity_preds = np.linalg.inv(H) @ conic_infinity @ np.linalg.inv(H.T)

    if np.allclose(H @ C_infinity @ H.T, conic_infinity):
        print("Transformation is correct ")

    output, similarity_rectified_transormation = MyWarp(image_affine_rectified, H, True)
    print('Select points for cosine evaluation other than previously selected points')
    cv2.imshow("input_image_cosine", image_copy_2)
    line_points_cosine = []
    print("Select first 4 points for first set of parallel lines")
    # selecting points for first two parallel lines
    cv2.setMouseCallback('input_image_cosine', select_points, (image_copy_2, line_points_cosine))
    while(1):
        cv2.imshow('input_image_cosine', image_copy_2)
        k = cv2.waitKey(10) & 0xFF
        if k == 27 or len(line_points_cosine)>=4:
            break
    assert(len(line_points_cosine) == 4)

    # selecting for second line
    print("Select another 4 points for second set of parallel lines")
    # cv2.imshow("input_image", image)
    cv2.setMouseCallback('input_image_cosine', select_points, (image_copy_2, line_points_cosine, 'r'))
    while(1):
        cv2.imshow('input_image_cosine', image_copy_2)
        k = cv2.waitKey(10) & 0xFF
        if k == 27 or len(line_points_cosine)>=8:
            break
    assert(len(line_points_cosine) == 8)
    # with open(annotations_file_name, 'w') as f:
    annotated_input_image_cosine = draw_annotations(image_copy_2, line_points_cosine)

    cosine_marked = annotated_rectified_image(similarity_rectified_transormation @ 
                                              affine_rectfied_transformation, 
                                              np.copy(output), 
                                              line_points_cosine)

    cv2.imwrite(annotated_input_image, annotated_image)
    cv2.imwrite(annotated_intermediate_filename, annotations_on_affine_rectified_image)
    cv2.imwrite(output_file_path, output)
    cv2.imwrite(cosine_annotated_filename, cosine_marked)
    cv2.imwrite(cosine_on_input_filename, annotated_input_image_cosine)
    cv2.imshow("Test lines annotated", cosine_marked)
    cv2.imshow("Similarity Rectified image", output)
    cv2.imshow("Test lines on input image", annotated_input_image_cosine)
    cv2.waitKey(0)
    result_info = find_cosine_before_and_after(line_points_cosine, 
                                                similarity_rectified_transormation @\
                                                affine_rectfied_transformation)
    result_info['affine_rectified_H'] = affine_rectfied_transformation.tolist()
    result_info['similarity_rectified_H'] = similarity_rectified_transormation.tolist()
    result_info['annotations_on_input_image'] = line_points
    with open(results_info_filename, 'w') as f:
        json.dump(result_info, f, indent=3)
   

if __name__ == '__main__':
    q2()