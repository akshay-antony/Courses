import cv2
from matplotlib import image 
import numpy as np
from utils import to_homogenous, normalize
from q1_a1 import make_constraints, select_points, plot_epipolar
import json
import os
import argparse
from tqdm import tqdm


def custom_matching(desc1, 
                    desc2,
                    tolerence=10):
    matches = []
    taken_desc2 = set()
    for i, d1 in tqdm(enumerate(desc1), total=desc1.shape[0]):
        curr_min = np.inf
        min_j = None
        for j, d2 in enumerate(desc2):
            curr_norm = (np.linalg.norm(d1-d2, 1))/d1.shape[0]
            if curr_min > curr_norm:
                min_j = j
                curr_min = curr_norm
        if curr_min <= tolerence:
            if not min_j in taken_desc2:
                taken_desc2.add(min_j)
                matches.append([i, min_j])
    print("matches", len(matches))
    return np.int16(matches)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', default="cuc")
    parser.add_argument('--tolerence', default=5)
    args = parser.parse_args()
    image_name = args.image_name
    image_1_filename = f"./data/q5/{image_name}_1.jpg"
    image_2_filename = f"./data/q5/{image_name}_2.jpg"
    image_1 = cv2.imread(image_1_filename)
    image_2 = cv2.imread(image_2_filename)
    image_1_gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    image_2_gray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    annotations_filename = f"./output/q5/{image_name}.json"
    result_1_filename = f"./output/q5/{image_name}_annotated_point.jpg"
    result_2_filename = f"./output/q5/{image_name}_annotated_line.jpg"
    result_filename = f"./output/q5/{image_name}.jpg"
    result_kp_filename = f"./output/q5/{image_name}_kp.jpg"

    sift = cv2.SIFT_create()
    kp_1, desc1 = sift.detectAndCompute(image_1_gray, None)
    # image_1_with_kp = cv2.drawKeypoints(image_1, kp_1, image_1)
    kp_2, desc2 = sift.detectAndCompute(image_2_gray, None)
    # image_2_with_kp = cv2.drawKeypoints(image_2, kp_2, image_2)
    
    pts1_tot = cv2.KeyPoint_convert(kp_1)
    pts2_tot = cv2.KeyPoint_convert(kp_2)

    # print(desc1, desc2)
    matches = custom_matching(desc1, desc2, float(args.tolerence))
    pts1_req = pts1_tot[matches[:, 0]]
    pts2_req = pts2_tot[matches[:, 1]]

    pts1_req = np.int16(pts1_req)
    pts2_req = np.int16(pts2_req)
    print(pts1_req.shape, pts2_req.shape)
    image_1_copy1 = np.copy(image_1)
    image_2_copy1 = np.copy(image_2)

    for pt_no, (pt1, pt2) in enumerate(zip(pts1_req, pts2_req)):
        image_1_copy1 = cv2.circle(image_1_copy1,
                                   pt1,
                                   radius=5,
                                   color=(0, 0, 255), 
                                   thickness=-1)
        image_2_copy1 = cv2.circle(image_2_copy1,
                                   pt2,
                                   radius=5,
                                   color=(0, 255, 0),
                                   thickness=-1)
        image_1_copy1 = cv2.putText(image_1_copy1,
                                    str(pt_no+1),
                                    pt1,
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    1,
                                    cv2.LINE_AA)
        image_2_copy1 = cv2.putText(image_2_copy1,
                                    str(pt_no+1),
                                    pt2,
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 255),
                                    1,
                                    cv2.LINE_AA)
    cv2.imshow("key points 1", image_1_copy1)
    cv2.imshow("key points 2", image_2_copy1)

    T1 = normalize(pts1_req)
    T2 = normalize(pts2_req)
    pts1_homo = to_homogenous(pts1_req)
    pts2_homo = to_homogenous(pts2_req)
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
    image_annotated, annotated_line = plot_epipolar(image_1, image_2, 
                                                annotations_filename, F)
    # cv2.imshow("key points 1", image_1_copy1)
    # cv2.imshow("key points 2", image_2_copy1)
    cv2.imshow("image 1", image_annotated)
    cv2.imshow("image 2", annotated_line)
    unq_no = len(os.listdir("./output/q5/"))
    print(unq_no, " of files contained")
    kp_concat = cv2.hconcat([image_1_copy1, image_2_copy1])
    concat_image = cv2.hconcat([image_annotated, annotated_line])
    cv2.imwrite(result_filename, concat_image)
    cv2.imwrite(result_kp_filename, kp_concat)
    # cv2.imwrite(result_1_filename.split('jpg')[0] + str(unq_no) + '.jpg', image_annotated)
    # cv2.imwrite(result_2_filename.split('jpg')[0] + str(unq_no) + '.jpg', annotated_line)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()