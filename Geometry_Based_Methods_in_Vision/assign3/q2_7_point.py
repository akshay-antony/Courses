import numpy as np
import cv2
import json
import os
from utils import normalize, to_homogenous,\
                  plot_epipolar, print_np_array
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

def seven_point(pts1_req, 
                pts2_req, 
                T1, 
                T2,
                pts1_tot_homo,
                pts2_tot_trans,
                tolerence):
    A = make_constraints(pts2_req, pts1_req)
    _, _, vt = np.linalg.svd(A)
    f1 = vt[-1, :]
    f2 = vt[-2, :]
    a = np.reshape(f1, (3, 3))
    b = np.reshape(f2, (3, 3))
    ## a equation
    coeffs = calc_7_point_coeff(a, b)
    outs = np.roots(coeffs)

    highest = 0
    F_best = None
    best_inlier_idx = None

    for out in outs:
        if np.isreal(out):
            root = np.real(out)
            lamb = root
            F = a*lamb + (1-lamb)*b
            F = T2.T@F@T1
            F /= F[-1, -1]
            corres_line = (F@pts1_tot_homo.T).T
            corres_line /= corres_line[:, -1].reshape((-1, 1))
            T_line = normalize(corres_line[:, :2])
            corres_line_trans = (T_line@corres_line.T).T

            x2fx1 = (corres_line_trans*pts2_tot_trans).sum(axis=1)
            inliers = np.sum(np.abs(x2fx1) <= tolerence)
            inliers_perc = inliers / pts1_tot_homo.shape[0]
            if inliers_perc > highest:
                highest = inliers_perc
                best_inlier_idx = np.abs(x2fx1) <= tolerence
                F_best = F
    return F_best, best_inlier_idx, highest

def calc_7_point_coeff(a, b):
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
    return coeffs

def make_constraints(pts1, pts2):
    x1, y1, z1 = pts1[:, 0].reshape(-1, 1), pts1[:, 1].reshape(-1, 1), pts1[:, 2].reshape(-1, 1)
    x2, y2, z2 = pts2[:, 0].reshape(-1, 1), pts2[:, 1].reshape(-1, 1), pts2[:, 2].reshape(-1, 1)
    A = np.concatenate((x1*x2, x1*y2, x1*z2, 
                        y1*x2, y1*y2, y1*z2,
                        x2*z1, z1*y2, z1*z2), axis=1)
    return A

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', default="teddy")
    parser.add_argument('--folder_name', default="q1a")
    parser.add_argument('--tolerence', default=6e-1)
    args = parser.parse_args()
    image_name = args.image_name
    folder_name = args.folder_name
    image_1_filename = f"./data/{folder_name}/{image_name}/image_1.jpg" 
    image_2_filename = f"./data/{folder_name}/{image_name}/image_2.jpg" 
    correspondence_filename = f"./data/{folder_name}/{image_name}/{image_name}_corresp_raw.npz"
    annotations_filename = f"./output/q1/results/{image_name}.json"
    results_info_filename = f'./output/q2b/results/{image_name}.json'
    result_1_filename = f'./output/q2b/{image_name}_1.jpg'
    result_2_filename = f'./output/q2b/{image_name}_2.jpg'
    result_filename = f'./output/q2b/{image_name}.jpg'
    image_1 = cv2.imread(image_1_filename)
    image_2 = cv2.imread(image_2_filename)
    correspondences = np.load(correspondence_filename)
    pts1 = correspondences['pts1']
    pts2 = correspondences['pts2']
    results_info = {}
    T1 = normalize(pts1)
    T2 = normalize(pts2)
    pts1_homo = to_homogenous(pts1)
    pts2_homo = to_homogenous(pts2)
    pts1_trans = (T1@pts1_homo.T).T
    pts2_trans = (T2@pts2_homo.T).T
    # check this

    iteration_limit = 1000_00
    tolerence = float(args.tolerence)
    iteration_no = 0
    highest = 0
    percentages = []
    opt_idx = None

    for iteration_no in tqdm(range(iteration_limit)):
        req_idx = np.random.randint(0, pts1.shape[0], (7))
        pts2_req = pts2_trans[req_idx]
        pts1_req = pts1_trans[req_idx]
        F_curr,\
        curr_best_inlier_idx,\
        curr_highest = seven_point(pts1_req,
                                   pts2_req,
                                   T1, 
                                   T2,
                                   pts1_homo,
                                   pts2_trans,
                                   tolerence)
        if highest < curr_highest:
            highest = curr_highest
            print(iteration_no, highest)
            opt_idx = curr_best_inlier_idx
            F_prelim_final = F_curr
        percentages.append(highest)
    
    pts2_req = pts2_trans[opt_idx]
    pts1_req = pts1_trans[opt_idx]
    F_final, _, _ = seven_point(pts1_req,
                                pts2_req,
                                T1,
                                T2,
                                pts1_homo,
                                pts2_trans,
                                tolerence)
    print(f"F for {image_name}: ")
    print_np_array(F_final, 6)
    print(np.linalg.matrix_rank(F_final))
    annotated_image, annotated_lines = plot_epipolar(image_1,
                                                     image_2,
                                                     annotations_filename,
                                                     F_final)
    fig, ax = plt.subplots(1)
    ax.set_xlabel('no of iterations')
    ax.set_ylabel('inlier percentage')
    ax.set_title(f'Progress for {image_name} image with tolerence {tolerence}')
    ax.plot(np.arange(iteration_limit).tolist(), percentages)
    results_info['F'] = F_final.tolist()
    results_info['tolerence'] = tolerence
    results_info['iteration_limit'] = iteration_limit
    results_info['inlier_percent'] = highest
    with open(results_info_filename, 'w') as f:
        json.dump(results_info, f, indent=2)
    fig.savefig(f'./output/q2b/results/plot_{image_name}.png')
    cv2.imshow("left image chosen point", annotated_image)
    cv2.imshow("right image epipolar line", annotated_lines) 
    # cv2.waitKey(0)
    concat_image = cv2.hconcat([annotated_image, annotated_lines])
    cv2.imwrite(result_filename, concat_image)
    cv2.imshow("concatenated", concat_image)
    # plt.show()
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        exit

if __name__ ==  '__main__':
    main()