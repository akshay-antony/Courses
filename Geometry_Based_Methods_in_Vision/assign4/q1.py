import numpy as np
import cv2
import argparse
from utils import to_homogenous, normalize, make_rotation_orthogonal
from tqdm import tqdm
import matplotlib.pyplot as plt
import json


def eight_point_with_ransac(pts1, pts2):
    T1 = normalize(pts1)
    T2 = normalize(pts2)
    pts1_tot_homo = to_homogenous(pts1)
    pts2_tot_homo = to_homogenous(pts2)
    pts1_tot_trans = (T1@pts1_tot_homo.T).T
    pts2_tot_trans = (T2@pts2_tot_homo.T).T
    
    iteration_limit = 100000
    tolerence = 0.6
    iteration_no = 0
    highest = 0
    F_req = None
    percentages = []
    opt_idx = None

    for iteration_no in tqdm(range(iteration_limit)):
        req_idx = np.random.randint(0, pts1.shape[0], (8))
        pts2_req = pts2[req_idx]
        pts1_req = pts1[req_idx]
        pts1_homo = to_homogenous(pts1_req)
        pts2_homo = to_homogenous(pts2_req)
        pts1_req = (T1@pts1_homo.T).T
        pts2_req = (T2@pts2_homo.T).T
        
        A = make_constraints(pts2_req, pts1_req)
        _, _, vt = np.linalg.svd(A)
        F = np.reshape(vt[-1, :], (3, 3))
        u, s, vt = np.linalg.svd(F)
        s = np.asarray([[s[0], 0, 0],
                        [0, s[1], 0],
                        [0, 0, 0]])
        F = u@s@vt
        F = T2.T@F@T1
        F /= F[-1, -1]
        inliers = 0
        corres_line = (F@pts1_tot_homo.T).T # N*3
        corres_line /= corres_line[:, -1].reshape((-1, 1))
        T_line = normalize(corres_line[:, :2])
        corres_line_trans = (T_line@corres_line.T).T
        x2fx1 = (corres_line_trans*pts2_tot_trans).sum(axis=1)
   
        inliers = np.sum(np.abs(x2fx1) <= tolerence)
        inliers_perc = inliers / pts1_tot_homo.shape[0]
        if highest < inliers_perc:
            highest = inliers_perc
            opt_idx = np.abs(x2fx1) <= tolerence
            print(iteration_no, highest)
            # F = T2.T@F@T1
            # F /= F[-1, -1]
            # F_req = F
        percentages.append(highest)
    pts2_req = pts2[opt_idx]
    pts1_req = pts1[opt_idx]
    pts1_homo = to_homogenous(pts1_req)
    pts2_homo = to_homogenous(pts2_req)
    pts1_req = (T1@pts1_homo.T).T
    pts2_req = (T2@pts2_homo.T).T
    A = make_constraints(pts2_req, pts1_req)
    _, _, vt = np.linalg.svd(A)
    F = np.reshape(vt[-1, :], (3, 3))
    u, s, vt = np.linalg.svd(F)
    s = np.asarray([[s[0], 0, 0],
                    [0, s[1], 0],
                    [0, 0, 0]])
    F = u@s@vt
    F = T2.T@F@T1
    F_req = F / F[-1, -1]
    return F_req

def find_four_solutions(U, VT):
    # check the eigen value of SVD(E)
    W = np.asarray([[0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1]])
    u3 = U[-1, :]
    R1 = U@W@VT 
    t1 = u3
    R2 = U@W@VT
    t2 = -u3
    R3 = U@W.T@VT
    t3 = u3
    R4 = U@W.T@VT
    t4 = -u3
    P1_ext = np.concatenate((R1, t1.reshape(3, 1)), axis=1)
    P2_ext = np.concatenate((R2, t2.reshape(3, 1)), axis=1)
    P3_ext = np.concatenate((R3, t3.reshape(3, 1)), axis=1)
    P4_ext = np.concatenate((R4, t4.reshape(3, 1)), axis=1)

    return P1_ext, P2_ext, P3_ext, P4_ext

def make_constraints(pts1, pts2):
    x1, y1, z1 = pts1[:, 0].reshape(-1, 1), pts1[:, 1].reshape(-1, 1), pts1[:, 2].reshape(-1, 1)
    x2, y2, z2 = pts2[:, 0].reshape(-1, 1), pts2[:, 1].reshape(-1, 1), pts2[:, 2].reshape(-1, 1)
    A = np.concatenate((x1*x2, x1*y2, x1*z2, 
                        y1*x2, y1*y2, y1*z2,
                        x2*z1, z1*y2, z1*z2), axis=1)
    return A

def make_triangulation_constraints(x1, x2, P1, P2):
    x1_mat = np.asarray([[0, -1, x1[1]],
                         [1, 0, -x1[0]],
                         [-x1[1], x1[0], 0]])
    x2_mat = np.asarray([[0, -1, x2[1]],
                         [1, 0, -x2[0]],
                         [-x2[1], x2[0], 0]])
    const_1 = x1_mat@P1
    const_2 = x2_mat@P2
    return np.concatenate((const_1[:2], const_2[:2]), axis=0)

def transform(total_points, H):
    if H.shape[0] != 4:
        H = np.concatenate((H, 
                            np.asarray([0, 0, 0, 1]).reshape((1, 4))), 
                            axis=0)
    return (H@to_homogenous(total_points).T).T

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name', 
                        type=str,
                        default="./data/monument/")
    args = parser.parse_args()
    image_1_filename = args.folder_name + "im1.jpg"
    image_2_filename = args.folder_name + "im2.jpg"
    keypoints_filename = args.folder_name + "some_corresp_noisy.npz"
    intrinsics_filename = args.folder_name + "intrinsics.npy"
    results_filename = "./output/q1/results.json"

    image_1 = cv2.imread(image_1_filename)
    print(image_1.shape)
    image_2 = cv2.imread(image_2_filename)
    keypoints = np.load(keypoints_filename, allow_pickle=True)
    intrinsics = np.load(intrinsics_filename, allow_pickle=True).item()
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']
    pts1 = keypoints['pts1']
    pts2 = keypoints['pts2']
    print("K1", K1)
    print("K2", K2)
    print(pts1.max(0), image_1.shape, K1)
    F = eight_point_with_ransac(pts1, pts2)
    E = K2.T@F@K1
    U_new, s_new, VT_new = np.linalg.svd(E)
    # print(f'U: {U_new} \n S: {s_new} \n V: {VT_new}')
    scale_req = np.sqrt(np.mean(s_new[:2]))
    s_new = np.asarray([[scale_req, 0, 0],
                        [0, scale_req, 0],
                        [0, 0, 0]])
    E_new = U_new@s_new@VT_new
    U_new, sample, VT_new =  np.linalg.svd(E_new) #* scale_req
    # print(sample)
    P2_exts = find_four_solutions(U_new, VT_new)
    P1_ext = np.concatenate((np.identity(3), np.zeros((3, 1))), axis=1)
    P1 = K1@P1_ext
    highest_z = 0
    baseline_recons = None
    R_final = None
    P_final = None

    for p_no, P2_ext in enumerate(P2_exts):
        P2 = K2@P2_ext
        total_points = np.zeros((0, 3))
        colors = np.zeros((0, 3))
        for i in tqdm(range(pts1.shape[0])):
            constraints = make_triangulation_constraints(pts1[i],
                                                         pts2[i],
                                                         P1,
                                                         P2)
            _, _, vt = np.linalg.svd(constraints)
            X = vt[-1, :]
            X /= X[-1]
        
            total_points = np.concatenate((total_points, 
                                           X[:3].reshape(1, -1)), axis=0)
            color = np.asarray([255, 0, 0])
            color = np.float16(color)/255
            colors = np.concatenate((colors, color.reshape(1, -1)), axis=0)
        colors[:, [0, 2]] = colors[:, [2, 0]]
        
        transformed_pts = transform(total_points, P2_ext)
        cam_2_perc = np.sum(transformed_pts[:, 2] > 0) / total_points.shape[0]
        cam_1_perc = np.sum(total_points[:, 2] > 0) / total_points.shape[0]

        if cam_2_perc + cam_1_perc > highest_z:
            baseline_recons = total_points
            highest_z = cam_1_perc + cam_2_perc
            R_final = make_rotation_orthogonal(P2_ext[:3, :3])
            P_final = P2_ext
            print(f"percent z {np.sum(transformed_pts[:, 2] > 0)/total_points.shape[0]}",
                  f"{np.sum(total_points[:, 2] > 0)/total_points.shape[0]}")
  
        R = make_rotation_orthogonal(P2_ext[:3, :3])     
        fig, ax = plt.subplots()
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(total_points[:, 0], 
                   total_points[:, 1],
                   total_points[:, 2],
                   color=colors[:])
        plt.show()
    fig, ax = plt.subplots()
    ax = fig.add_subplot(111, projection = '3d')
    colors[:, [0, 2]] = colors[:, [2, 0]]
    ax.scatter(baseline_recons[:, 0], 
               baseline_recons[:, 1],
               baseline_recons[:, 2],
               color=colors[:])
    plt.show()
    results_dict = {}
    results_dict['P2'] = P_final.tolist()
    results_dict['R'] = R_final.tolist()
    with open(results_filename, 'w') as f:
        json.dump(results_dict, f, indent=3)
    np.save("./output/q1/monument.npy", baseline_recons)
        # print(np.sum(total_points[:, 1] > 0)/total_points.shape[0])
    # print(np.linalg.svd(E_new))

if __name__ == '__main__':
    main()