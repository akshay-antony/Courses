import numpy as np
import cv2 

def make_rotation_orthogonal(R):
    U, _, VT = np.linalg.svd(R)
    new_R = U@VT
    if np.linalg.det(new_R) <= 0:
        new_R = U@np.diag(np.asarray([1, 1, -1]))@VT
    # assert(new_R@new_R.T) == np.identity(3))
    return new_R

def to_homogenous(x):
    return np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)

def normalize(points):
    '''
    points of size n*2
    '''
    x0, y0 = np.mean(points[:, 0]), np.mean(points[:, 1])
    d_avg = np.mean(np.sqrt((points[:, 0] - x0)**2 +
                            (points[:, 1] - y0)**2))
    s = np.sqrt(2) / d_avg
    T = np.asarray([[s, 0, -s*x0],
                    [0, s, -s*y0],
                    [0, 0, 1]])
    return T

if __name__ == '__main__':
    pass