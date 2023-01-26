import numpy as np
import cv2
from random import randint


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
        if len(params) == 3 and params[2] == 'r':
            cv2.circle(image, (x, y), 0, (0, 0, 255), 10)
        else:
            cv2.circle(image, (x, y), 0, (0, 255, 0), 10)
        line_points.append([x, y])

def select_points_homography(event, 
                             x, 
                             y, 
                             flags, 
                             params):
    image = params[0]
    # list to append point to
    points = params[1]
    if event == cv2.EVENT_LBUTTONUP:
        print(x, y)
        cv2.circle(image, (x, y), 0, (0, 0, 255), 10)
        cv2.putText(image, 
                    str(len(points)+1), 
                    (x, y), 
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=1,
                    color=(0, 0, 255), )
                    #thickness=10)
        points.append([x, y])


def normalize(v):
    return v / np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def normalize_projective(v):
    return v / v[-1]

def MyWarp(img, H, return_Ht=False):
    h, w = img.shape[:2]
    pts = np.array([[0,0],[0,h],[w,h],[w,0]], dtype=np.float64).reshape(-1,1,2)
    pts = cv2.perspectiveTransform(pts, H)
    [xmin, ymin] = (pts.min(axis=0).ravel() - 0.5).astype(int)
    [xmax, ymax] = (pts.max(axis=0).ravel() + 0.5).astype(int)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])

    result = cv2.warpPerspective(img, Ht.dot(H), (xmax-xmin, ymax-ymin))
    if return_Ht:
        return result, Ht.dot(H)
    return result

def cosine(u, v):
    return (u[0] * v[0] + u[1] * v[1]) / (np.sqrt(u[0]**2 + u[1]**2) * np.sqrt(v[0]**2 + v[1]**2))


def find_cosine_before_and_after(line_points, H):
    line_points_transformed = []
    for line_point in line_points:
        transformed_point = H @ np.asarray([[line_point[0]],
                                            [line_point[1]],
                                            [1]])
        transformed_point = normalize_projective(transformed_point)
        line_points_transformed.append([transformed_point[0][0], transformed_point[1][0]])

    # calculating cosines
    cosines_after = []
    cosines_before = []
    for i in range(len(line_points)//4):
        # assume lines ai + bj, ci + dj
        a = line_points[4*i+1][0] - line_points[4*i][0]
        b = line_points[4*i+1][1] - line_points[4*i][1]
        c = line_points[4*i+3][0] - line_points[4*i+2][0]
        d = line_points[4*i+3][1] - line_points[4*i+2][1]
        cosine_value_before = cosine([a, b], [c, d])
        cosines_before.append(cosine_value_before)
    
        a = line_points_transformed[4*i+1][0] - line_points_transformed[4*i][0]
        b = line_points_transformed[4*i+1][1] - line_points_transformed[4*i][1]
        c = line_points_transformed[4*i+3][0] - line_points_transformed[4*i+2][0]
        d = line_points_transformed[4*i+3][1] - line_points_transformed[4*i+2][1]
        cosine_value_after = cosine([a, b], [c, d])
        print(f"Cosine before: {cosine_value_before}, "+ \
              f"Cosine after: {cosine_value_after}")
        cosines_after.append(cosine_value_after)
    
    cosines_dict = {'cosines_before': cosines_before,
                    'cosines_after': cosines_after}
    return cosines_dict

def draw_annotations(image, line_points):
    for i in range(len(line_points) // 2):
        if(i%2 == 0):
            r = randint(0, 255)
            g = randint(0, 255)
            b = randint(0, 255)
            rand_color = (r, g, b)
        cv2.line(image, 
                 (int(line_points[2*i][0]), int(line_points[2*i][1])),
                 (int(line_points[2*i+1][0]), int(line_points[2*i+1][1])),
                 color=rand_color,
                 thickness=5)
    return image

def dynamic_annotation(image, line_points):
    # draws the current pair of lines
    r = randint(0, 255)
    g = randint(0, 255)
    b = randint(0, 255)
    rand_color = (r, g, b)
    for i in range(len(line_points) // 2):
        cv2.line(image, 
                (int(line_points[2*i][0]), int(line_points[2*i][1])),
                (int(line_points[2*i+1][0]), int(line_points[2*i+1][1])),
                color=rand_color,
                thickness=5)
    return image

def draw_annotations_homography(image, points):
    for i, point in enumerate(points):
        cv2.circle(image, (point[0], point[1]), 5,  (0, 0, 255), -1)
        cv2.putText(image, 
                str(i+1), 
                (point[0], point[1]), 
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1,
                color=(0, 0, 255), )
    return image

def standardize(data, mean_req=(0, 0), std_req=(np.sqrt(2), np.sqrt(2))):
    if isinstance(data, list):
        data = np.asarray(data) # N*2
    mean_initial = np.mean(data, axis=0) 
    std_initial = np.std(data, axis=0)
    ## transformation matrix
    H = np.asarray(
            [[std_req[0]/std_initial[0], 0, mean_req[0]-(std_req[0]/std_initial[0])*mean_initial[0]],
            [0, std_req[1]/std_initial[1], mean_req[1]-(std_req[1]/std_initial[1])*mean_initial[1]],
            [0, 0, 1]]
        )
    return H

def to_homogenous(data, scale=1):
    if isinstance(data, list):
        data = np.asarray(data)
    data = np.concatenate((data, np.ones((data.shape[0], 1))*scale), axis=1)
    assert(data.shape[1] == 3)
    return data

def batch_transformation(H, points):
    # funtion to transform a batch of points
    if len(points.shape) == 2:
        points = points.reshape(-1, points.shape[1], 1)
        transformed_points = np.einsum('MN, BNi -> BMi', H, points)
    return transformed_points.reshape(-1, points.shape[1])

def warp_for_homography(img, H, dim, return_Ht=False):
    h, w = dim
    pts = np.array([[0,0],[0,h],[w,h],[w,0]], dtype=np.float64).reshape(-1,1,2)
    pts = cv2.perspectiveTransform(pts, H)
    [xmin, ymin] = (pts.min(axis=0).ravel() - 0.5).astype(int)
    [xmax, ymax] = (pts.max(axis=0).ravel() + 0.5).astype(int)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])

    result = cv2.warpPerspective(img, H, 
                                 (dim[1], dim[0]), 
                                 cv2.INTER_CUBIC,)
    if return_Ht:
        return result, Ht.dot(H)
    return result

def make_points_cyclic(points):
    if isinstance(points, list):
        points = np.asarray(points)

    centroid = np.mean(points)
    points_rel = points - centroid
    angle = np.arctan2(points_rel[:, 1], points_rel[:, 0])
    angle_order = np.argsort(angle)
    return angle_order

def annotated_rectified_image(H, img, line_points):
    transformed_points = batch_transformation(H, to_homogenous(line_points))
    transformed_points /= transformed_points[:, -1].reshape(-1, 1)
    transformed_points = np.int16(transformed_points).tolist()
    return draw_annotations(img, transformed_points)

