import numpy as np
import cv2
import argparse
from utils import to_homogenous, make_rotation_orthogonal
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib 
import matplotlib.cm as cmx
import os
from q1 import make_triangulation_constraints
import json


def construct_cam_matrix(K, R, T):
    extrinsic = np.concatenate((R, T.reshape(3, 1)), axis=1)
    return K@extrinsic

def color_by_z(points, colors):
    colors_predef = np.asarray([[255, 0, 0],
                                [255, 255, 0],
                                [0, 255, 0],
                                [0, 0, 255]])/255
    colors_predef = np.flip(colors_predef, 0)
    z_min = np.min(points[:, -1])
    z_max = np.max(points[:, -1])
    interval = (z_max - z_min)/4
    for i in range(4):
        curr_min = z_min + i*interval
        curr_max = z_min + (i+1)*interval
        curr_idx = (points[:, -1] >= curr_min) & \
                   (points[:, -1] <= curr_max)
        colors[curr_idx, :] = colors_predef[i]
    return colors

def triangulate(pts1, 
                pts2, 
                P1, 
                P2, 
                image,
                total_points=None,
                colors=None):
    if total_points is None:
        total_points = np.zeros((0, 3))
    if colors is None:
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
        color = image[int(pts1[i, 1]), int(pts1[i, 0]), :]
        color = np.float16(color)/255
        colors = np.concatenate((colors, color.reshape(1, -1)), axis=0)
    colors[:, [0, 2]] = colors[:, [2, 0]]
    return total_points, colors

def plot_3d(points, colors):
    fig, ax = plt.subplots()
    ax = fig.add_subplot(111, projection = '3d')
    cmhot = plt.get_cmap("turbo")
    z_normalized = points[:, -1] - np.min(points[:, -1]) / \
                   (np.max(points[:, -1]) - np.min(points[:, -1]))
    ax.scatter(points[:, 0], 
               points[:, 1],
               points[:, 2],
               c=z_normalized,
               cmap=cmhot
            )
    plt.show()

def find_matches_bf(pts1, pts2):
    result = []

    for i, pt1 in tqdm(enumerate(pts1), total=pts1.shape[0]):
        for j, pt2 in enumerate(pts2):
            if pt1[0] == pt2[0] and pt2[1] == pt1[1]:
                # print(pt1, pt2)
                result.append([i, j])
    return np.asarray(result)

def find_matches_optmized(pts1, pts2):
    result = []
    pts2_dict = {}
    for i, pt in enumerate(pts2):
        pts2_dict[f"{pt[0]},{pt[1]}"] = i

    for i, pt1 in enumerate(pts1):
        curr_str = f"{pt1[0]},{pt1[1]}"
        if curr_str in pts2_dict:
            result.append([i, pts2_dict[curr_str]])
    return np.asarray(result)


def find_unknown_matches_bf(pts1, pts2):
    '''
    pts1: the one we are going to filter
    pts2: based on which filtering is done
    '''
    pts2_set = set()
    for pt in pts2:
        pts2_set.add(f"{pt[0]},{pt[1]}")
    result = []

    for i, pt1 in tqdm(enumerate(pts1), total=pts1.shape[0]):
        if not f"{pt1[0]},{pt1[1]}" in pts2_set:
            result.append(i)
    return np.asarray(result)

def pnp(image_coords, point_coords):
    image_coords = to_homogenous(image_coords)
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
    return camera_matrix

def find_R_t(camera_matrix,  K_org):
    R_t = np.linalg.inv(K_org)@camera_matrix
    print(R_t)
    return R_t[:3, :3], R_t[:, -1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name',
                        default="./data/data_cow/")
    args = parser.parse_args()
    image_folder_name = os.path.join(args.folder_name, "images")
    camera_folder_name = os.path.join(args.folder_name, "cameras")
    corresp_folder_name = os.path.join(args.folder_name, "correspondences")
    image_names_list = [os.path.join(image_folder_name, filename)
                            for filename in os.listdir(image_folder_name)]
    images = [cv2.imread(image_filename)
                for i, image_filename in enumerate(image_names_list)]
    image_1 = images[0]
    image_2 = images[1]
    image_3 = images[2]
    image_4 = images[3]
    results_filename = "./output/q2/results_info.json"
    results_info = {}

    # for cam 1, 2
    corresp_12_cam1 = np.load(os.path.join(
                                corresp_folder_name,
                                "./pairs_1_2/cam1_corresp.npy"))
    corresp_12_cam2 = np.load(os.path.join(
                                corresp_folder_name,
                                "./pairs_1_2/cam2_corresp.npy"))
    # for cam 1, 2, 3
    corresp_13_cam1 = np.load(os.path.join(
                            corresp_folder_name,
                            "./pairs_1_3/cam1_corresp.npy"))                          
    corresp_13_cam2 = np.load(os.path.join(
                              corresp_folder_name,
                              "./pairs_1_3/cam2_corresp.npy"))
    corresp_23_cam1 = np.load(os.path.join(
                              corresp_folder_name,
                              "./pairs_2_3/cam1_corresp.npy"))                          
    corresp_23_cam2 = np.load(os.path.join(
                              corresp_folder_name,
                              "./pairs_2_3/cam2_corresp.npy"))
    
    # for cam 1, 2, 3, 4
    corresp_14_cam1 = np.load(os.path.join(
                            corresp_folder_name,
                            "./pairs_1_4/cam1_corresp.npy"))                          
    corresp_14_cam2 = np.load(os.path.join(
                              corresp_folder_name,
                              "./pairs_1_4/cam2_corresp.npy"))
    corresp_24_cam1 = np.load(os.path.join(
                              corresp_folder_name,
                              "./pairs_2_4/cam1_corresp.npy"))                          
    corresp_24_cam2 = np.load(os.path.join(
                              corresp_folder_name,
                              "./pairs_2_4/cam2_corresp.npy"))
    corresp_34_cam1 = np.load(os.path.join(
                              corresp_folder_name,
                              "./pairs_3_4/cam1_corresp.npy"))                          
    corresp_34_cam2 = np.load(os.path.join(
                              corresp_folder_name,
                              "./pairs_3_4/cam2_corresp.npy"))
    ###

    cam1_params = np.load(os.path.join(camera_folder_name,
                                       "cam1.npz"),
                                       allow_pickle=True)
    cam2_params = np.load(os.path.join(camera_folder_name,
                                      "cam2.npz"))
    cam1_matrix = construct_cam_matrix(cam1_params['K'],
                                       cam1_params['R'],
                                       cam1_params['T'])
    cam2_matrix = construct_cam_matrix(cam2_params['K'],
                                       cam2_params['R'],
                                       cam2_params['T'])
    total_points, colors = triangulate(corresp_12_cam1, 
                                       corresp_12_cam2,
                                       cam1_matrix,
                                       cam2_matrix,
                                       image_1)
    total_points_cam_12 = np.copy(total_points)
    matches_13_common = np.int16(find_matches_optmized(
                                    corresp_12_cam1, 
                                    corresp_13_cam1))
    matches_unknown_13_common = np.int16(find_unknown_matches_bf(
                                            corresp_13_cam1,
                                            corresp_12_cam1))
    matches_unknown_23_common = np.int16(find_unknown_matches_bf(
                                            corresp_23_cam1,
                                            corresp_12_cam2))
    
    cam3_matrix = pnp(corresp_13_cam2[matches_13_common[:, 1], :], 
                      total_points[matches_13_common[:, 0], :])
    results_info["cam3"] = np.round(cam3_matrix, 4).tolist()
    R3, t3 = find_R_t(cam3_matrix, cam1_params['K'])
    R3_orthogonal = make_rotation_orthogonal(R3)
    results_info["cam3_R_before_orthogonal"] = np.round(R3, 4).tolist()
    results_info["cam3_R"] = np.round(R3_orthogonal, 4).tolist()
    results_info["cam3_t"] = np.round(t3, 4).tolist()
    
    plot_3d(total_points, colors)
    print(f"cam 1 and 2 gives {total_points.shape}")
    np.save("./output/q2/cam_1_2.npy", total_points)
    matches_13_required_cam1 = corresp_13_cam1[matches_unknown_13_common]
    matches_13_required_cam2 = corresp_13_cam2[matches_unknown_13_common]
    matches_23_required_cam1 = corresp_23_cam1[matches_unknown_23_common]
    matches_23_required_cam2 = corresp_23_cam2[matches_unknown_23_common]
    # print(f"uncommon 13 {matches_13_required_cam1}, 13 {matches_13_required_cam2}")
    # print(f"uncommon 23 {matches_23_required_cam1}, 23 {matches_23_required_cam2}")

    total_points, colors = triangulate(
                                matches_13_required_cam1,
                                matches_13_required_cam2,
                                cam1_matrix,
                                cam3_matrix,
                                image_1,
                                total_points,
                                colors)
    total_points, colors = triangulate(
                                matches_23_required_cam1,
                                matches_23_required_cam2,
                                cam2_matrix,
                                cam3_matrix,
                                image_2,
                                total_points,
                                colors)
    plot_3d(total_points, colors)
    print(f"cam 1, 2 and 3 gives {total_points.shape}")
    np.save("./output/q2/cam_1_2_3.npy", total_points)

    ###
    # registering 4th cam
    matches_14_common = np.int16(find_matches_optmized(
                                    corresp_12_cam1, 
                                    corresp_14_cam1))
    print(matches_14_common.shape, total_points_cam_12.shape)
    cam4_matrix = pnp(corresp_14_cam2[matches_14_common[:, 1], :], 
                      total_points_cam_12[matches_14_common[:, 0], :])
    
    results_info["cam4"] = np.round(cam4_matrix, 4).tolist()
    R4, t4 = find_R_t(cam4_matrix, cam1_params['K'])
    R4_orthogonal = make_rotation_orthogonal(R4)
    results_info["cam4_R_before_orthogonal"] = np.round(R4, 4).tolist()
    results_info["cam4_R"] = np.round(R4_orthogonal, 4).tolist()
    results_info["cam4_t"] = np.round(t4, 4).tolist()
    ## find unreconstructed points in cam 1, present in cam 4
    matches_unknown_14_common = np.int16(find_unknown_matches_bf(
                                    corresp_14_cam1,
                                    np.concatenate((
                                        corresp_12_cam1,
                                        corresp_13_cam1),0)))
    matches_unknown_24_common = np.int16(find_unknown_matches_bf(
                                    corresp_24_cam1,
                                    np.concatenate((
                                        corresp_12_cam2,
                                        corresp_23_cam1), 0)))
    matches_unknown_34_common = np.int16(find_unknown_matches_bf(
                                    corresp_34_cam1,
                                    np.concatenate((
                                        corresp_13_cam2,
                                        corresp_23_cam2), 0)))
    matches_req_cam_4 = np.concatenate((matches_unknown_14_common,
                                        matches_unknown_24_common,
                                        matches_unknown_34_common), 0)
    matches_req_cam_4 = np.unique(matches_req_cam_4, axis=0)
    ###
    total_points, colors = triangulate(corresp_24_cam1[matches_unknown_24_common],
                                            corresp_24_cam2[matches_unknown_24_common],
                                            cam2_matrix,
                                            cam4_matrix,
                                            image_2,
                                            total_points,
                                            colors)
    total_points, colors = triangulate(corresp_34_cam1[matches_unknown_34_common],
                                            corresp_34_cam2[matches_unknown_34_common],
                                            cam3_matrix,
                                            cam4_matrix,
                                            image_3,
                                            total_points,
                                            colors)
    plot_3d(total_points, colors)
    print(f"cam 1, 2, 3 and 4 gives {total_points.shape}")
    np.save("./output/q2/cam_1_2_3_4.npy", total_points)
    print(matches_unknown_14_common.shape,
          matches_unknown_24_common.shape,
          matches_unknown_34_common.shape)
    # print(matches_13_common.shape, total_points.shape, corresp_12_cam1.shape)
    with open(results_filename, "w") as f:
        json.dump(results_info, f, indent=3)
    

if __name__ == '__main__':
    main()