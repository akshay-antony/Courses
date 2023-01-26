import numpy as np
import cv2 
import json
from utils import to_homogenous, normalize
from tqdm import tqdm
import matplotlib.pyplot as plt
from q3 import make_constraints
from scipy.optimize import least_squares

def objective(x, pts1, pts2, P1):
    '''
    x of size = (12 + 4*n)
    pts1, pts2 n*2
    '''
    # P1 = np.asarray([[1, 0, 0, 0],
    #                  [0, 1, 0, 0],
    #                  [0, 0, 1, 1]])
    P2 = np.reshape(x[0:12], (3, 4))
    X = x[12:].reshape((-1, 4)) # n*4
    X /= X[:, -1].reshape((-1, 1))
    x1 = (X@P1.T)
    x2 = (X@P2.T)
    x1 /= x1[:, -1].reshape((-1, 1))
    x2 /= x2[:, -1].reshape((-1, 1))
    # print(x1, x2)
    x1 = x1[:, :2]
    x2 = x2[:, :2]
    d = np.concatenate((pts1-x1, pts2-x2), axis=1).reshape(-1)
    d_error = np.mean(np.concatenate((pts1-x1, pts2-x2), axis=1)**2)
    print(d_error)
    return d

def triangulate(P1, P2, pts1, pts2, image_1):
    total_points = np.zeros((0, 3))
    colors = np.zeros((0, 3))
    total_points_initial = np.zeros((0, 4))
    P1 /= P1[-1, -1]
    P2 /= P2[-1, -1]
    # initial solution
    for i in tqdm(range(pts1.shape[0])):
        constraints = make_constraints(pts1[i],
                                       pts2[i],
                                       P1,
                                       P2)
        _, _, vt = np.linalg.svd(constraints)
        X = vt[-1, :]
        X /= X[-1]
        total_points_initial = np.concatenate((total_points_initial, X.reshape((1, -1))), axis=0)
        total_points = np.concatenate((total_points, 
                                       X[:3].reshape(1, -1)), axis=0)
        color = image_1[pts1[i, 1], pts1[i, 0], :]
        color = np.float16(color)/255
        colors = np.concatenate((colors, color.reshape(1, -1)), axis=0)
    return total_points, colors

def main():
    image_1_filename = "./data/q4/img1.jpg"
    image_2_filename = "./data/q4/img2.jpg"
    P1_filename = "./data/q4/P1_noisy.npy"
    P2_filename = "./data/q4/P2_noisy.npy"
    pts1_filename = "./data/q4/pts1.npy"
    pts2_filename = "./data/q4/pts2.npy"
    image_1 = cv2.imread(image_1_filename)
    image_2 = cv2.imread(image_2_filename)
    P1 = np.load(P1_filename)
    P2 = np.load(P2_filename)
    pts1 = np.load(pts1_filename)
    pts2 = np.load(pts2_filename)
    total_points = np.zeros((0, 3))
    colors = np.zeros((0, 3))
    total_points_initial = np.zeros((0, 4))
    print(pts1.shape)
    # initial solution
    for i in tqdm(range(pts1.shape[0])):
        constraints = make_constraints(pts1[i],
                                       pts2[i],
                                       P1,
                                       P2)
        _, _, vt = np.linalg.svd(constraints)
        X = vt[-1, :]
        X /= X[-1]
        total_points_initial = np.concatenate((total_points_initial, 
                                               X.reshape((1, -1))), axis=0)
        total_points = np.concatenate((total_points, 
                                       X[:3].reshape(1, -1)), axis=0)
        color = image_1[pts1[i, 1], pts1[i, 0], :]
        color = np.float16(color)/255
        colors = np.concatenate((colors, color.reshape(1, -1)), axis=0)

    print(4*total_points.shape[0])
    # p1_flatten = P1.reshape(-1)
    p2_flatten = P2.reshape(-1)
    # p2_flatten = np.asarray([[1, 0, 0, 0],
    #                          [0, 1, 0, 0],
    #                          [0, 0, 1, 1]]).reshape(-1)
    total_points_initial_flatten = total_points_initial.reshape(-1)
    inital_x = np.concatenate((p2_flatten,
                               total_points_initial_flatten), axis=0)
    print(inital_x.shape)
    colors[:, [0, 2]] = colors[:, [2, 0]]
    fig, ax = plt.subplots()
    ax = fig.add_subplot(111, projection = '3d')
    sample_rate = 1
    # ax.scatter(total_points[::sample_rate, 0], 
    #            total_points[::sample_rate, 1],
    #            total_points[::sample_rate, 2],
    #            color=colors[::sample_rate])
    # plt.show()

    ## doing optimization
    res = least_squares(objective, 
                        inital_x, 
                        args=(pts1, pts2, P1))
    print(res)
    
    soln = res.x
    print(soln.shape)
    P1_new = soln[:12].reshape((3, 4))
    P2_new = soln[12: 24].reshape((3, 4))
    # calculating using new values
    X_new = soln[12:].reshape((-1, 4))
    #X_new, colors = triangulate(P1_new, P2_new, pts1, pts2, image_1)
    X_new /= X_new[:, -1].reshape((-1, 1))
    ax.scatter(X_new[::sample_rate, 0], 
               X_new[::sample_rate, 1],
               X_new[::sample_rate, 2],
               color=colors[::sample_rate]) 
    plt.show()

if __name__ == '__main__':
    main()