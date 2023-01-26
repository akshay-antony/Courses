from email.mime import image
import numpy as np
import os 
import cv2 
from q1_a1 import make_constraints, select_points
from utils import to_homogenous, normalize, plot_epipolar, print_np_array
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', default='toybus')
    args = parser.parse_args()
    image_name = args.image_name
    image_1_filename = f"./data/q1b/{image_name}/image_1.jpg" 
    image_2_filename = f"./data/q1b/{image_name}/image_2.jpg" 
    correspondence_filename = f"./data/q1b/{image_name}/{image_name}_7_point_corresp.npz"
    annotations_filename = f"./output/q1/results/{image_name}.json"
    result_1_filename = f'./output/q1/{image_name}_1.jpg'
    result_2_filename = f'./output/q1/{image_name}_2.jpg'
    result_filename = f'./output/q1/{image_name}.jpg'
    image_1 = cv2.imread(image_1_filename)
    image_2 = cv2.imread(image_2_filename)
    correspondences = np.load(correspondence_filename)
    pts1 = correspondences['pts1']
    pts2 = correspondences['pts2']

    T1 = normalize(pts1)
    T2 = normalize(pts2)
    pts1_homo = to_homogenous(pts1)
    pts2_homo = to_homogenous(pts2)
    pts1_trans = (T1@pts1_homo.T).T
    pts2_trans = (T2@pts2_homo.T).T
    A = make_constraints(pts2_trans, pts1_trans)
    _, _, vt = np.linalg.svd(A)
    f1 = vt[-1, :]
    f2 = vt[-2, :]
    a = np.reshape(f1, (3, 3))
    b = np.reshape(f2, (3, 3))

    ## a equation
    coeff_lambda_3 = a[0, 0]*a[1, 1]*a[2, 2] - a[0, 0]*a[1, 2]*a[2, 1] - a[0, 1]*a[1, 0]*a[2, 2] + \
                     a[0, 1]*a[1, 2]*a[2, 0] + a[0, 2]*a[1, 0]*a[2, 1] - a[0, 2]*a[1, 1]*a[2, 0] - \
                     a[0, 0]*a[1, 1]*b[2, 2] + a[0, 0]*a[1, 2]*b[2, 1] + a[0, 0]*a[2, 1]*b[1, 2] - \
                     a[0, 0]*a[2, 2]*b[1, 1] + a[0, 1]*a[1, 0]*b[2, 2] - a[0, 1]*a[1, 2]*b[2, 0] - \
                     a[0, 1]*a[2, 0]*b[1, 2] + a[0, 1]*a[2, 2]*b[1, 0] - a[0, 2]*a[1, 0]*b[2, 1] + \
                     a[0, 2]*a[1, 1]*b[2, 0] + a[0, 2]*a[2, 0]*b[1, 1] - a[0, 2]*a[2, 1]*b[1, 0] - \
                     a[1, 0]*a[2, 1]*b[0, 2] + a[1, 0]*a[2, 2]*b[0, 1] + a[1, 1]*a[2, 0]*b[0, 2] - \
                     a[1, 1]*a[2, 2]*b[0, 0] - a[1, 2]*a[2, 0]*b[0, 1] + a[1, 2]*a[2, 1]*b[0, 0] + \
                     a[0, 0]*b[1, 1]*b[2, 2] - a[0, 0]*b[1, 2]*b[2, 1] - a[0, 1]*b[1, 0]*b[2, 2] + \
                     a[0, 1]*b[1, 2]*b[2, 0] + a[0, 2]*b[1, 0]*b[2, 1] - a[0, 2]*b[1, 1]*b[2, 0] - \
                     a[1, 0]*b[0, 1]*b[2, 2] + a[1, 0]*b[0, 2]*b[2, 1] + a[1, 1]*b[0, 0]*b[2, 2] - \
                     a[1, 1]*b[0, 2]*b[2, 0] - a[1, 2]*b[0, 0]*b[2, 1] + a[1, 2]*b[0, 1]*b[2, 0] + \
                     a[2, 0]*b[0, 1]*b[1, 2] - a[2, 0]*b[0, 2]*b[1, 1] - a[2, 1]*b[0, 0]*b[1, 2] + \
                     a[2, 1]*b[0, 2]*b[1, 0] + a[2, 2]*b[0, 0]*b[1, 1] - a[2, 2]*b[0, 1]*b[1, 0] - \
                     b[0, 0]*b[1, 1]*b[2, 2] + b[0, 0]*b[1, 2]*b[2, 1] + b[0, 1]*b[1, 0]*b[2, 2] - \
                     b[0, 1]*b[1, 2]*b[2, 0] - b[0, 2]*b[1, 0]*b[2, 1] + b[0, 2]*b[1, 1]*b[2, 0]
    coeff_lambda_2 = a[0, 0]*a[1, 1]*b[2, 2] - a[0, 0]*a[1, 2]*b[2, 1] - a[0, 0]*a[2, 1]*b[1, 2] + \
                     a[0, 0]*a[2, 2]*b[1, 1] - a[0, 1]*a[1, 0]*b[2, 2] + a[0, 1]*a[1, 2]*b[2, 0] + \
                     a[0, 1]*a[2, 0]*b[1, 2] - a[0, 1]*a[2, 2]*b[1, 0] + a[0, 2]*a[1, 0]*b[2, 1] - \
                     a[0, 2]*a[1, 1]*b[2, 0] - a[0, 2]*a[2, 0]*b[1, 1] + a[0, 2]*a[2, 1]*b[1, 0] + \
                     a[1, 0]*a[2, 1]*b[0, 2] - a[1, 0]*a[2, 2]*b[0, 1] - a[1, 1]*a[2, 0]*b[0, 2] + \
                     a[1, 1]*a[2, 2]*b[0, 0] + a[1, 2]*a[2, 0]*b[0, 1] - a[1, 2]*a[2, 1]*b[0, 0] - \
                     2*a[0, 0]*b[1, 1]*b[2, 2] + 2*a[0, 0]*b[1, 2]*b[2, 1] + 2*a[0, 1]*b[1, 0]*b[2, 2] - \
                     2*a[0, 1]*b[1, 2]*b[2, 0] - 2*a[0, 2]*b[1, 0]*b[2, 1] + 2*a[0, 2]*b[1, 1]*b[2, 0] + \
                     2*a[1, 0]*b[0, 1]*b[2, 2] - 2*a[1, 0]*b[0, 2]*b[2, 1] - 2*a[1, 1]*b[0, 0]*b[2, 2] + \
                     2*a[1, 1]*b[0, 2]*b[2, 0] + 2*a[1, 2]*b[0, 0]*b[2, 1] - 2*a[1, 2]*b[0, 1]*b[2, 0] - \
                     2*a[2, 0]*b[0, 1]*b[1, 2] + 2*a[2, 0]*b[0, 2]*b[1, 1] + 2*a[2, 1]*b[0, 0]*b[1, 2] - \
                     2*a[2, 1]*b[0, 2]*b[1, 0] - 2*a[2, 2]*b[0, 0]*b[1, 1] + 2*a[2, 2]*b[0, 1]*b[1, 0] + \
                     3*b[0, 0]*b[1, 1]*b[2, 2] - 3*b[0, 0]*b[1, 2]*b[2, 1] - 3*b[0, 1]*b[1, 0]*b[2, 2] + \
                     3*b[0, 1]*b[1, 2]*b[2, 0] + 3*b[0, 2]*b[1, 0]*b[2, 1] - 3*b[0, 2]*b[1, 1]*b[2, 0]
    coeff_lambda_1 = a[0, 0]*b[1, 1]*b[2, 2] - a[0, 0]*b[1, 2]*b[2, 1] - a[0, 1]*b[1, 0]*b[2, 2] + \
                     a[0, 1]*b[1, 2]*b[2, 0] + a[0, 2]*b[1, 0]*b[2, 1] - a[0, 2]*b[1, 1]*b[2, 0] - \
                     a[1, 0]*b[0, 1]*b[2, 2] + a[1, 0]*b[0, 2]*b[2, 1] + a[1, 1]*b[0, 0]*b[2, 2] - \
                     a[1, 1]*b[0, 2]*b[2, 0] - a[1, 2]*b[0, 0]*b[2, 1] + a[1, 2]*b[0, 1]*b[2, 0] + \
                     a[2, 0]*b[0, 1]*b[1, 2] - a[2, 0]*b[0, 2]*b[1, 1] - a[2, 1]*b[0, 0]*b[1, 2] + \
                     a[2, 1]*b[0, 2]*b[1, 0] + a[2, 2]*b[0, 0]*b[1, 1] - a[2, 2]*b[0, 1]*b[1, 0] - \
                     3*b[0, 0]*b[1, 1]*b[2, 2] + 3*b[0, 0]*b[1, 2]*b[2, 1] + 3*b[0, 1]*b[1, 0]*b[2, 2] - \
                     3*b[0, 1]*b[1, 2]*b[2, 0] - 3*b[0, 2]*b[1, 0]*b[2, 1] + 3*b[0, 2]*b[1, 1]*b[2, 0]
    coeff_lambda_0 =  b[0, 0]*b[1, 1]*b[2, 2] - b[0, 0]*b[1, 2]*b[2, 1] - b[0, 1]*b[1, 0]*b[2, 2] + \
                      b[0, 1]*b[1, 2]*b[2, 0] + b[0, 2]*b[1, 0]*b[2, 1] - b[0, 2]*b[1, 1]*b[2, 0]
    coeffs = np.asarray([coeff_lambda_3, coeff_lambda_2, coeff_lambda_1, coeff_lambda_0])
    outs = np.roots(coeffs)
    print(outs)
    for out in outs:
        if np.isreal(out):
            root = np.real(out)
            break
    lamb = root
    # lamb = 0
    F = a*lamb + (1-lamb)*b
    F = T2.T@F@T1
    F /= F[-1, -1]
    print_np_array(F, 8)
    print(np.linalg.matrix_rank(F))
    image_annotated, annotated_line = plot_epipolar(image_1,
                                                    image_2,
                                                    annotations_filename,
                                                    F) 
    cv2.imshow("image 1", image_annotated)
    cv2.imshow("image 2", annotated_line)
    concat_image = cv2.hconcat([image_annotated, annotated_line])
    cv2.imwrite(result_1_filename, image_annotated)
    cv2.imwrite(result_2_filename, annotated_line)
    cv2.imwrite(result_filename, concat_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
