## Assignment 3
### To run q1_a1 for 8 point
```python3 q1_a1.py --image_name=teddy```, the only argument to program is image name without extension Image is assumed to present in /data/q1a/
### To run q1_a2.py to find essential matrix
```python3 q1_a2.py --image_name=teddy```, the only argument to program is image name without extension. Image is assumed to present in /data/q1a/
### To run q1_b.py for 7 point algorithm
```python3 q1_b.py --image_name=toytrain```, the only argument to program is image name without extension. Image is assumed to present in /data/q1b/
### To run q2_8_point.py for 8 point with RANSAC
```python3 q2_8_point.py --image_name=toytrain --folder_name=q1b --tolerence=6e-1 ```, the arguments need the image name without extension and the folder name in which image should be found. This is so because toytrain and chair are present in different folders. Tolerence is the value below which we consider two correspondences as an inlier
### To run q2_7_point.py for 7 point with RANSAC
```python3 q2_7_point.py --image_name=toytrain --folder_name=q1b --tolerence=6e-1 ```, the arguments need the image name without extension and the folder name in which image should be found. This is so because toytrain and chair are present in different folders. Tolerence is the value below which we consider two correspondences as an inlier.
### To run q3.py for triangulation
```python3 q3.py ```. No arguments are required because we are only solving for 1 image, so paths are hard coded
### To run q4.py for bundle adjustment
```python3 q4.py ```. No arguments are required because we are only solving for 1 image, so paths are hard coded
### To run q5.py for custom fundamental matrix estimation
 ```python3 q5.py --image_name=cuc --tolerence=4.5```. The argument is the image name, and expects two corresponding images to be found at ./data/q5/{image_name}_1.jpg and ./data/q5/{image_name}_2.jpg, tolerence is l1 loss less that which two sift descriptors are considered match.