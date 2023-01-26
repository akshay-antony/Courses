import numpy as np
import open3d as o3d
import os
import time
from scipy.spatial.transform import Rotation as Rot


def render(pcd, pcd_np):
    total_no = 500
    angle_req = 2*np.pi/total_no
    angles = np.linspace(0, 2*np.pi, 300)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    angle_no = 0
    while True:
        r = Rot.from_euler('zxy', 
                           [0, 0, angle_req],
                           degrees=False)
        R = r.as_matrix()
        pcd_np = (R@pcd_np.T).T
        pcd.points = o3d.utility.Vector3dVector(pcd_np)
        #vis.update(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)
        angle_no += 1
        if angle_no >= total_no:
            break

def main():
    pcd1_path = "./output/q2/cam_1_2.npy"
    pcd2_path = "./output/q2/cam_1_2_3.npy"
    pcd3_path = "./output/q2/cam_1_2_3_4.npy"
    pcd1_np = np.load(pcd1_path)
    pcd1_np[:, 1] = -pcd1_np[:, 1]
    pcd2_np = np.load(pcd2_path)
    pcd2_np[:, 1] = -pcd2_np[:, 1] 
    pcd3_np = np.load(pcd3_path)
    pcd3_np[:, 1] = -pcd3_np[:, 1]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd1_np)
    #3d.visualization.draw_geometries([pcd])
    
    
    
    # render(pcd, pcd1_np)
    # pcd.points = o3d.utility.Vector3dVector(pcd2_np)
    # render(pcd, pcd2_np)
    # pcd.points = o3d.utility.Vector3dVector(pcd3_np)
    # render(pcd, pcd3_np)

    ## for monument
    pcd_path = "./output/q1/monument.npy"
    pcd_np = np.load(pcd_path)
    pcd_np[:, 1] = -pcd_np[:, 1] 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    render(pcd, pcd_np)
    

if __name__ == '__main__':
    main()