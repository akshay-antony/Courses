import numpy as np
import cv2 
import json
from utils import to_homogenous, normalize
from tqdm import tqdm
import matplotlib.pyplot as plt


def objective(x, pts1, pts2):
    '''
    x of size = (12 + 12 + 4*n)
    pts1, pts2 n*2
    '''
    P1 = np.reshape(x[:12], (3, 4))
    P2 = np.reshape(x[12:24], (3, 4))
    X = x[24:].reshape((-1, 4)) # n*4
    X /= X[:, -1].reshape((-1, 1))
    x1 = (X@P1.T)
    x2 = (X@P2.T)
    x1 /= x1[:, -1].reshape((-1, 1))
    x2 /= x2[:, -1].reshape((-1, 1))
    # print(x1, x2)
    x1 = x1[:, :2]
    x2 = x2[:, :2]
    d = np.sum((pts1-x1)**2 + (pts2-x2)**2)
    # print(d)
    return d

def make_constraints(x1, x2, P1, P2):
    x1_mat = np.asarray([[0, -1, x1[1]],
                         [1, 0, -x1[0]],
                         [-x1[1], x1[0], 0]])
    x2_mat = np.asarray([[0, -1, x2[1]],
                         [1, 0, -x2[0]],
                         [-x2[1], x2[0], 0]])
    const_1 = x1_mat@P1
    const_2 = x2_mat@P2
    return np.concatenate((const_1[:2], const_2[:2]), axis=0)

def main():
    image_1_filename = "./data/q3/img1.jpg"
    image_2_filename = "./data/q3/img2.jpg"
    P1_filename = "./data/q3/P1.npy"
    P2_filename = "./data/q3/P2.npy"
    pts1_filename = "./data/q3/pts1.npy"
    pts2_filename = "./data/q3/pts2.npy"
    image_1 = cv2.imread(image_1_filename)
    image_2 = cv2.imread(image_2_filename)
    P1 = np.load(P1_filename)
    P2 = np.load(P2_filename)
    pts1 = np.load(pts1_filename)
    pts2 = np.load(pts2_filename)
    total_points = np.zeros((0, 3))
    total_points_initial = np.zeros((0, 4))
    colors = np.zeros((0, 3))

    for i in tqdm(range(pts1.shape[0])):
        constraints = make_constraints(pts1[i],
                                       pts2[i],
                                       P1,
                                       P2)
        _, _, vt = np.linalg.svd(constraints)
        X = vt[-1, :]
        X /= X[-1]
        total_points_initial = np.concatenate((total_points_initial,
                                               X.reshape(1, -1)), axis=0)
        total_points = np.concatenate((total_points, 
                                       X[:3].reshape(1, -1)), axis=0)
        color = image_1[pts1[i, 1], pts1[i, 0], :]
        color = np.float16(color)/255
        colors = np.concatenate((colors, color.reshape(1, -1)), axis=0)

    p1_flatten = P1.reshape(-1)
    p2_flatten = P2.reshape(-1)
    X_flatten = total_points_initial.reshape(-1)
    x = np.concatenate((p1_flatten, 
                        p2_flatten,
                        X_flatten), axis=0)
    d = objective(x, pts1, pts2)
    print(d, pts1.shape)
    colors[:, [0, 2]] = colors[:, [2, 0]]
    fig, ax = plt.subplots()
    ax = fig.add_subplot(111, projection = '3d')
    sample_rate = 1
    ax.scatter(total_points[::sample_rate, 0], 
               total_points[::sample_rate, 1],
               total_points[::sample_rate, 2],
               color=colors[::sample_rate])
    plt.show()

if __name__ == '__main__':
    main()