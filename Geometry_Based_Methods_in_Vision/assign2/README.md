# HW2: Single-view Reconstruction
## Q1
The program q1.py implements DLT with minimum 6 3D-2D correspondences
```python3 q1.py```

The argument needed is the image name without extension, and txt file with 3d-2d correspondences, and npy file with 3d points which should be projected to the image

## Q2
### a
For calculating K we need three vanishing points in orthogonal directions. To find them user should annotated 3 sets of parallel lines (end points are annoated, lines are obtained from cross product) in input image that are in orthogonal directions in 3D. The vanishing points are found as cross product of these annoated lines. To run the program set the image_name (q2a.png) in program which will load image from data folder and the program runs.

```python3 q2.py```

### b
For calculating K here, 3 sqaures are annotated in input image as 4 cyclic points. Three homographies that unrectifies a unit square to these 3 squares are calculated using DLT, and consraints for SVD are made. The SVD is solved to obtain K, and equations of normals of each square is calculated using K, and 2 vanishing points of a square and they are measured to know the correctness of the solution. The only argument is image_name which should be image name without extension (q2b.png).

```python3 q2b.py```

## Q3
### a 
For calculating the intrinsix matrix, code from q2a is used. After that planes are annotated by selecting 4 points in a cyclic manner. Equation of each plane is calculated from a reference point and normals calculated from vanishing points. From equation of the plane, scales of all pixels are calculated. The only argument is the image_name (q3.png).

```python3 q3.py```

### b
change image_name inside main function, and run
```python3 q3.py```
