from traceback import print_tb
from weakref import ref
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import json
from annotations import vis_annotations_given_points
from q2 import find_vanishing_point, select_points, make_constraints
from utils import plot_planes, to_homogenous, angle_between_planes
from q2b import find_orientation
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from tqdm import tqdm


def select_reference(event, 
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
    image_name = "posner"
    image_filename = f"./data/{image_name}.png"
    input_image = cv2.imread(image_filename)
    annotations_filename = f"./output/q3/annotations/{image_name}.json"
    annotated_image_filename = f"./output/q3/annotations/{image_name}.jpg"
    image_copy = np.copy(input_image)
    image_copy_2 = np.copy(input_image)
    image_copy_3 = np.copy(input_image)
    image_copy_4 = np.copy(input_image)
    image_copy_5 = np.copy(input_image)
    annotations_dict = {}

    choice = input("Do you want to use saved K (y/n): ")
    line_points = []
    if choice == "n":
        for i in range(3):
            cv2.imshow("input_image", input_image)
            print("Select 4 points for set parallel lines")
            # selecting points for first two parallel lines
            if i == 0:
                color = (0, 0, 255)
            elif i == 1:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            
            cv2.setMouseCallback('input_image', select_points, (image_copy, line_points, color))
            while(1):
                cv2.imshow('input_image', image_copy)
                k = cv2.waitKey(10) & 0xFF
                if k == 27 or len(line_points) == (i+1)*4:
                    break
        annotations_dict['vanishing_points'] = line_points
    else:
        with open(annotations_filename, 'r') as f:
            line_points = json.load(f)['vanishing_points']

    annotations_dict['vanishing_points'] = line_points
    vis_annotations_given_points(line_points, input_image)
    vanishing_points = []
    for i in range(len(line_points)//4):
        vanishing_point = find_vanishing_point(line_points[4*i: 4*i+4])
        vanishing_points.append(vanishing_point)
    vanishing_points = np.asarray(vanishing_points).reshape(-1, 3)
    vanishing_points /= vanishing_points[:, 2].reshape(-1, 1)
    A = np.zeros((0, 4))
    A = make_constraints(vanishing_points[0], vanishing_points[1], A)
    A = make_constraints(vanishing_points[0], vanishing_points[2], A)
    A = make_constraints(vanishing_points[1], vanishing_points[2], A)
    _, _, v = np.linalg.svd(A)
    v = v[-1, :]
    omega = np.asarray([[v[0], 0, v[1]],
                        [0, v[0], v[2]],
                        [v[1], v[2], v[3]]])
    L = np.linalg.cholesky(omega)
    K = np.linalg.inv(L.T)
    K /= K[-1, -1]
    print("K")
    for row in range(3):
        for col in range(3):
            print(K[row, col])

    choice = input("Do you want to use saved plane annotations (y/n): ")
    plane_annotations = []
    if choice == "n":
        cv2.imshow("input_image_plane", image_copy_2)
        selected_all = "n"
        i = 0
        while selected_all == "n":
            if i == 0:
                color = (0, 0, 255)
            elif i == 1:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            print("select 4 points for the plane")
            cv2.setMouseCallback('input_image_plane', select_points, (image_copy_2, plane_annotations, color))
            while(1):
                cv2.imshow('input_image_plane', image_copy_2)
                k = cv2.waitKey(10) & 0xFF
                if k == 27 or len(plane_annotations) == (i+1)*4:
                    break
            i += 1
            selected_all = input("Are you done selecting planes (y/n): ")
            if selected_all == "y":
                break
        annotations_dict['plane_annotations'] = plane_annotations      
    else:
        with open(annotations_filename, 'r') as f:
            plane_annotations = json.load(f)['plane_annotations']
    annotations_dict['plane_annotations'] = plane_annotations
    directions = []

    for i in range(len(plane_annotations)//4):
        image_copy_3 = plot_planes(plane_annotations[4*i: 4*i+4], 
                                   image_copy_3, i)
        direction = find_orientation(plane_annotations[4*i: 4*i+4], K)
        directions.append(direction)

    directions = np.asarray(directions)
    directions /= directions[:, 2].reshape((-1, 1))
    
    choice = input("Do you want to use saved ref point (y/n): ")
    reference_point = []
    if choice == "n":
        cv2.imshow("reference_image", image_copy_4)
        print("select reference point")
        cv2.setMouseCallback('reference_image', select_reference, 
                            (image_copy_4, reference_point, (0, 255, 0)))
        while(1):
            cv2.imshow('reference_image', image_copy_4)
            k = cv2.waitKey(10) & 0xFF
            if k == 27 or len(reference_point) == 1:
                break
        reference_depth = [1]
        annotations_dict['reference_point'] = reference_point
        annotations_dict['reference_depth'] = reference_depth
        with open(annotations_filename, 'w') as f:
            json.dump(annotations_dict, f, indent=4)
    else:
        with open(annotations_filename, 'r') as f:
            references = json.load(f)
            reference_point = references['reference_point']
            reference_depth = references['reference_depth']
    cv2.image = cv2.circle(image_copy_3, (reference_point[0][0], 
                           reference_point[0][1]), 10, (255, 0, 255), -1)
    cv2.imshow("planes", image_copy_3)
    cv2.imwrite(annotated_image_filename, image_copy_3)
    known_point_1 = np.asarray([reference_point[0][0], 
                                reference_point[0][1], 
                                reference_depth[0]])
    known_point_2 = np.asarray([519, 240, 1])
    P = np.concatenate((K, np.asarray([0, 0, 0]).reshape(3, 1)), axis=1)
    P_plus = P.T@np.linalg.inv(P@P.T)
    known_point_1 = P_plus @ known_point_1
    known_point_2 = P_plus @ known_point_2
    # print(lamb.shape, ref_point.shape, direction.shape, ref_point)
    print("dir", directions, "k", K)
    print("first", angle_between_planes(directions[0], directions[1]))
    print("second", angle_between_planes(directions[1], directions[2]))
    print("third", angle_between_planes(directions[2], directions[0]))
    total_points = np.zeros((0, 3))
    colors = np.zeros((0, 3))
    print("P_plus", P_plus)
    print("planes", plane_annotations)


    all_planes = []
    scalaras = []
    for plane_no in range(len(plane_annotations)//4):
        if plane_no >= 3:
            a = -directions[plane_no].T @ known_point_2[:3]
        else:
            a = -directions[plane_no].T @ known_point_1[:3]
        scalaras.append(a)
        polygon = Polygon(plane_annotations[4*plane_no: 4*plane_no+4])
        all_planes.append(polygon)

    sample_rate = 4
    print(image_copy_5.shape, plane_annotations[0: 4])
    for i in tqdm(range(image_copy_5.shape[0])):
        for j in range(image_copy_5.shape[1]):
            for plane_no, plane_poly in enumerate(all_planes):   
                if i%sample_rate != 0 or j %sample_rate != 0:
                    continue
                if plane_poly.contains(Point(j, i)):
                    # print(Point(j, i), " is in ", plane_no)
                    a = scalaras[plane_no]
                    curr_point = P_plus @ np.asarray([j, i, 1])
                    curr_point = curr_point[:3]
                    lamb = - a / (directions[plane_no].reshape(1, -2) @ curr_point.reshape(-1, 1))
                    curr_point = lamb * curr_point
                    total_points = np.concatenate((total_points, curr_point.reshape(1, -1)), axis=0)
                    color = np.asarray([image_copy_5[i, j, 2]/255, 
                                        image_copy_5[i, j, 1]/255,
                                        image_copy_5[i, j, 0]/255])
                    colors = np.concatenate((colors, color.reshape(1, -1)), axis=0)
                    # continue
    cv2.waitKey(0)
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