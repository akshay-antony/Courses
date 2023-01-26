import numpy as np
import cv2 
import json
import argparse
from utils import MyWarp, \
                  normalize,\
                  dynamic_annotation,\
                  select_points,\
                  find_cosine_before_and_after,\
                  draw_annotations, \
                  annotated_rectified_image, \
                  normalize_projective
from affine_rectification import find_affine_rectification_matrix


def q2(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', default='tile5',
                        help="name of the image to recitify")

    args = parser.parse_args(raw_args)

    # filenames to read from
    image_filename = f"./data/q1/{args.image_name}.jpg"

    annotated_input_image = f'./output/q4/input_image_annotated/{args.image_name}.jpg'
    output_file_path = f'./output/q4/similarity_rectified/{args.image_name}.jpg'
    results_info_filename = f'./output/q4/results_info/{args.image_name}.json'
    cosine_annotated_filename = f'./output/q4/cosine_marked/{args.image_name}.jpg'
    cosine_on_input_filename = f'./output/q4/cosines_on_input/{args.image_name}.jpg'

    input_image = cv2.imread(image_filename)
    image_copy_2 = np.copy(input_image)
    print("Using affine rectified image")
    input_image_copy = np.copy(input_image)

    choice = input("Do you want to use stored annotations (y/n): ")
    line_points = []

    if choice == 'n':
        cv2.imshow("input_image", input_image_copy)
        for annotation_no in range(5):
            print(f"Please select {annotation_no+1}th perpendicular line")
            cv2.setMouseCallback("input_image", select_points, (input_image_copy, line_points))
            input_image_copy = dynamic_annotation(input_image_copy, line_points[-4:])
            while(1):
                cv2.imshow('input_image', input_image_copy)
                k = cv2.waitKey(10) & 0xFF
                if k == 27 or len(line_points) == 4*(annotation_no+1):
                    break
            assert(len(line_points) == 4*(annotation_no+1))

    else:
        with open(results_info_filename, 'r') as f:
            line_points = json.load(f)['annotations_on_input_image']
            assert(len(line_points) >= 20) 

    line_points_homogenous = line_points#np.concatenate([np.asarray(line_points), 
                                             #np.ones((len(line_points), 1))], axis=1).reshape(-1, 3)
    
    annotated_image = draw_annotations(input_image_copy, line_points)
    while 1:
        cv2.imshow("Annotated perpendicular lines on input image", annotated_image)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break
    print(line_points_homogenous)
    # finding perpendicular lines
    line_equations = []
    for i in range(len(line_points_homogenous)//2):
        line_equation = np.cross(np.asarray([line_points_homogenous[2*i][0],
                                             line_points_homogenous[2*i][1],
                                             1]),
                                 np.asarray([line_points_homogenous[2*i+1][0], 
                                             line_points_homogenous[2*i+1][1],
                                             1]))
        line_equations.append(normalize(line_equation))
    
    A = np.zeros((len(line_equations)//2, 6))
    
    for i in range(len(line_equations) // 2):
        # A[3*i] = l[i][0]*m[i][0], 
        #          l[i][1]*m[i][0] + l[i][0]*m[i][1], 
        #          l[i][2]*m[i][0] + l[i][0]*m[i][2],
        #          l[i][1]*m[i][1],
        #          l[i][2]*m[i][1] + l[i][1]*m[i][2], 
        #          l[i][2]*m[i][2]
        print(2*i, 2*i+1)
        A[i, 0] = line_equations[2*i][0] * line_equations[2*i+1][0]
        A[i, 1] = line_equations[2*i][1] * line_equations[2*i+1][0] + line_equations[2*i][0] * line_equations[2*i+1][1]
        A[i, 2] = line_equations[2*i][2] * line_equations[2*i+1][0] + line_equations[2*i][0] * line_equations[2*i+1][2]
        A[i, 3] = line_equations[2*i][1] * line_equations[2*i+1][1]
        A[i, 4] = line_equations[2*i][2] * line_equations[2*i+1][1] + line_equations[2*i][1] * line_equations[2*i+1][2]
        A[i, 5] = line_equations[2*i][2] * line_equations[2*i+1][2]
    
    print(f"A's shape {A.shape}, {A}")
    _, singular_values, vh = np.linalg.svd(A)
    req_eigen_vector = vh[-1, :] / vh[-1, -1]
    print("all vec", req_eigen_vector)
    a, b_2, d_2, c, e_2, f = req_eigen_vector
    print(a, b_2, d_2, c, e_2, f)
    C_infinity = np.asarray([[a, b_2, d_2],
                             [b_2, c, e_2],
                             [d_2, e_2, f]])
    C_infinity[:, :] /= C_infinity[-1, -1]
    print(singular_values, C_infinity)
    u, s, uh = np.linalg.svd(C_infinity)
    # s[:] /= s[-1]
    print(u, s, uh)
    diagonal_singular = np.asarray([[1/np.sqrt(s[0]), 0, 0],
                                    [0, 1/np.sqrt(s[1]), 0],
                                    [0, 0, 1]]) 
    H = diagonal_singular @ uh
    H[:, :] /= H[-1, -1]
    conic_infinity = np.asarray([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]])
    C_infinity_preds = np.linalg.inv(H) @ conic_infinity @ np.linalg.inv(H.T)

    if np.allclose(H @ C_infinity @ H.T, conic_infinity):
        print("Transformation is correct ")

    output, similarity_rectified_transormation = MyWarp(input_image, H, True)
    print('Select points for cosine evaluation other than previously selected points')
    cv2.imshow("input_image_cosine", image_copy_2)
    line_points_cosine = []
    # print("Select first 4 points for first set of parallel lines")
    # # selecting points for first two parallel lines
    # cv2.setMouseCallback('input_image_cosine', select_points, (image_copy_2, line_points_cosine))
    # while(1):
    #     cv2.imshow('input_image_cosine', image_copy_2)
    #     k = cv2.waitKey(10) & 0xFF
    #     if k == 27 or len(line_points_cosine)>=4:
    #         break
    # assert(len(line_points_cosine) == 4)

    # selecting for second line
    # print("Select another 4 points for second set of parallel lines")
    # # cv2.imshow("input_image", image)
    # cv2.setMouseCallback('input_image_cosine', select_points, (image_copy_2, line_points_cosine, 'r'))
    # while(1):
    #     cv2.imshow('input_image_cosine', image_copy_2)
    #     k = cv2.waitKey(10) & 0xFF
    #     if k == 27 or len(line_points_cosine)>=8:
    #         break
    # assert(len(line_points_cosine) == 8)
    # with open(annotations_file_name, 'w') as f:
    annotated_input_image_cosine = draw_annotations(image_copy_2, line_points_cosine)

    # cosine_marked = annotated_rectified_image(similarity_rectified_transormation, 
    #                                           np.copy(output), 
    #                                           line_points_cosine)

    cv2.imwrite(annotated_input_image, annotated_image)
    cv2.imwrite(output_file_path, output)
    # cv2.imwrite(cosine_annotated_filename, cosine_marked)
    cv2.imwrite(cosine_on_input_filename, annotated_input_image_cosine)
    # cv2.imshow("Test lines annotated", cosine_marked)
    cv2.imshow("Similarity Rectified image", output)
    cv2.imshow("Test lines on input image", annotated_input_image_cosine)
    cv2.waitKey(0)
    result_info = find_cosine_before_and_after(line_points_cosine, 
                                                similarity_rectified_transormation)
    result_info['similarity_rectified_H'] = similarity_rectified_transormation.tolist()
    result_info['annotations_on_input_image'] = line_points
    with open(results_info_filename, 'w') as f:
        json.dump(result_info, f, indent=3)
   

if __name__ == '__main__':
    q2()