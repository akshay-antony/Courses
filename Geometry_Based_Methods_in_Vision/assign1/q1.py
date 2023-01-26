import cv2
import numpy as np
import json 
from utils import MyWarp, select_points,\
                  find_cosine_before_and_after,\
                  draw_annotations, annotated_rectified_image
from affine_rectification import find_affine_rectification_matrix
import argparse


def fit_line(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept
    
def q1(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', help='name of the image to input')
    args = parser.parse_args(raw_args)
    print(f"Running affine rectification on {args.image_name}")

    image_filename = f"./data/q1/{args.image_name}.jpg"
    annotations_file_name = f"./data/q1_annotations/{args.image_name}.json"
    image = cv2.imread(image_filename)
    choice = input("Load existing annotations(y/n): ")
    image_copy = np.copy(image)
    image_copy_2 = np.copy(image)
    input_annotated_filename = f"./output/q1/input_image_annotated/{args.image_name}.jpg"
    output_filename = f"./output/q1/affine_recitified/{args.image_name}.jpg"
    results_info_filename = f'./output/q1/results_info/{args.image_name}.json'
    cosine_annotated_filename = f'./output/q1/cosine_marked/{args.image_name}.jpg'
    cosine_on_input_filename = f'./output/q1/cosines_on_input/{args.image_name}.jpg'
    cv2.imshow("input_image", image_copy)

    if choice == 'y':
        with open(annotations_file_name, 'r') as f:
            line_points = json.load(f)
        assert(len(line_points) == 8)
            
    else:
        cv2.imshow("input_image", image_copy)
        line_points = []
        print("Select first 4 points for first set of parallel lines")
        
        # selecting points for first two parallel lines
        cv2.setMouseCallback('input_image', select_points, (image_copy, line_points))
        while(1):
            cv2.imshow('input_image', image_copy)
            k = cv2.waitKey(10) & 0xFF
            if k == 27 or len(line_points)>=4:
                break
        assert(len(line_points) == 4)

        # selecting for second line
        print("Select another 4 points for second set of parallel lines")
        # cv2.imshow("input_image", image)
        cv2.setMouseCallback('input_image', select_points, (image_copy, line_points, 'r'))
        while(1):
            cv2.imshow('input_image', image_copy)
            k = cv2.waitKey(10) & 0xFF
            if k == 27 or len(line_points)>=8:
                break
        assert(len(line_points) == 8)
        with open(annotations_file_name, 'w') as f:
            json.dump(line_points, f, indent=4)

    for i in range(4):
        cv2.line(image_copy, 
                 (line_points[2*i][0], line_points[2*i][1]),
                 (line_points[2*i+1][0], line_points[2*i+1][1]),
                 color=(0, 255, 0) if i < 2 else (0, 0, 255),
                 thickness=5)

    annotated_input_image = draw_annotations(image_copy, line_points)
    while 1:
        cv2.imshow("Annotated parallel lines on input image", image_copy)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break

    H = find_affine_rectification_matrix(line_points)
    output, affine_rectified_H = MyWarp(image, H, True)

    print('Select points for cosine evaluation other than previously selected points')
    cv2.imshow("input_image_cosine", image_copy_2)
    line_points_cosine = []
    print("Select first 4 points for first set of parallel lines")
    # selecting points for first two parallel lines
    cv2.setMouseCallback('input_image_cosine', select_points, (image_copy_2, line_points_cosine))
    while(1):
        cv2.imshow('input_image_cosine', image_copy_2)
        k = cv2.waitKey(10) & 0xFF
        if k == 27 or len(line_points_cosine)>=4:
            break
    assert(len(line_points_cosine) == 4)

    # selecting for second line
    print("Select another 4 points for second set of parallel lines")
    # cv2.imshow("input_image", image)
    cv2.setMouseCallback('input_image_cosine', select_points, (image_copy_2, line_points_cosine, 'r'))
    while(1):
        cv2.imshow('input_image_cosine', image_copy_2)
        k = cv2.waitKey(10) & 0xFF
        if k == 27 or len(line_points_cosine)>=8:
            break
    assert(len(line_points_cosine) == 8)
    # with open(annotations_file_name, 'w') as f:
    annotated_input_image_cosine = draw_annotations(image_copy_2, line_points_cosine)

    cosine_marked = annotated_rectified_image(affine_rectified_H, 
                                              np.copy(output), 
                                              line_points_cosine)
    cv2.imshow("Affine-Rectified Image", output)

    # calculating cosines
    results_info = find_cosine_before_and_after(line_points_cosine, H)
    results_info['affined_rectified_H'] = affine_rectified_H.tolist()
    results_info['parallel_annotations'] = line_points
    cv2.imshow("Test lines on Affine-Rectified Image", cosine_marked)
    with open(results_info_filename, 'w') as f:
        json.dump(results_info, f, indent=4)
    cv2.imwrite(input_annotated_filename, annotated_input_image)
    cv2.imwrite(output_filename, output)
    cv2.imwrite(cosine_annotated_filename, cosine_marked)
    cv2.imwrite(cosine_on_input_filename, annotated_input_image_cosine)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    q1()