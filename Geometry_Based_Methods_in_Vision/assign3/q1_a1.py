from matplotlib import image
import numpy as np
import cv2
import json
import argparse
from utils import normalize, \
                  to_homogenous, \
                  plot_epipolar, \
                  print_np_array

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
    parser.add_argument('--image_name', default='chair')
    args = parser.parse_args()
    image_name = args.image_name
    image_1_filename = f"./data/q1a/{image_name}/image_1.jpg" 
    image_2_filename = f"./data/q1a/{image_name}/image_2.jpg" 
    correspondence_filename = f"./data/q1a/{image_name}/{image_name}_corresp_raw.npz"
    annotations_filename = f"./output/q1/results/{image_name}.json"
    result_1_filename = f'./output/q1/{image_name}_1.jpg'
    result_2_filename = f'./output/q1/{image_name}_2.jpg'
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
    # check this
    A = make_constraints(pts2_trans, pts1_trans)
    _, _, vt = np.linalg.svd(A)
    F = np.reshape(vt[-1, :], (3, 3))
    u, s, vt = np.linalg.svd(F)
    s = np.asarray([[s[0], 0, 0],
                    [0, s[1], 0],
                    [0, 0, 0]])
    F = u@s@vt
    F = T2.T@F@T1
    F /= F[-1, -1]

    print(f"obtained F: ")
    print_np_array(F, 8)
    image_annotated, annotated_line = plot_epipolar(image_1, 
                                                    image_2, 
                                                    annotations_filename,
                                                    F)
 
    cv2.imshow("image 1", image_annotated)
    cv2.imshow("image 2", annotated_line)
    cv2.imwrite(result_1_filename, image_annotated)
    cv2.imwrite(result_2_filename, annotated_line)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()