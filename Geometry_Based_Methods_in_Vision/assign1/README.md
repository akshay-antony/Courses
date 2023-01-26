# Documentation of code
1. For affine rectification, run q1.py

    Arguments taken ```--image_name```. The name of image is such that the path ```./data/q1/{image_name}.jpg``` contains the image. It is assumed that image is in .jpg format.

    Example:  ```python3 q1.py --image_name=tiles4```

    All results are saved in a folder named ```./output/q1```

    After program begins, start annotating pair of parallel lines. To do this select 8 points, such that consecutive points form a line. For calculating affine rectification the funtion from ```affine_rectification.py``` is used. All lines and intersection of lines are calculated using cross product.

2. For metric rectification, run q2.py

    Arguments taken ```--image_name```. The name of image is such that the path ```./data/q1/{image_name}.jpg``` contains the image. It is assumed that image is in .jpg format.

    Example:  ```python3 q2.py --image_name=tiles4```

    All results are saved in a folder named ```./output/q2/```

    After program begins, start annotating pair of perpendicular lines. To do this select 8 points, such that consecutive points form a line (4 perpendicular lines). The program assumes you have saved annotations for affine rectification, because q2 loads those affine annotations to calculate the affine rectification matrix to first transform to affine rectified Image. For calculating affine rectification the funtion from ```affine_rectification.py``` is used

2. For homography, run q3.py

    Arguments taken ```--first_image_name```, ```--second_image_name```. The first argument is name of the normal image such that the path ```./data/q3/{image_name}``` contains first image. The second argument is name of the perspective image such that ```./data/q3/{image_name}``` contains the image. It is expected to give the image names with the extensions.

    Example:  ```python3 q3.py --first_image_name=desk-normal.png --second_image_name=desk-perspective.png```

    All results are saved in a folder named ```./output/q3/```

    After program begins, start annotating 4 points on perspective image in the following order (top left, top right, bottom left, bottom right). The points on normal image is assumed as the 4 edge points defined using width and height of the image.

4. For homography of multiple normal images, run q5.py

    Arguments taken are ```--perspective_image_name```
    which is the image name of the perspective image expected such that ```./data/q5/{perspective_image_name``` contains the image.
    ```--normal_image_names```, which is a list of normal image names separated by commas. Such that each image_name should be contained in ```./data/q5/```

    Example:  ```python3 q5.py --perspective_image_name=advertise_perspective.jpg --second_image_name=sixth-sense.jpg, prestige.jpg, memories.jpg```

    All results are saved in a folder named ```./output/q5/```

    After program begins, start annotating 4*n (where n is the number of normal images) points on perspective image in the following order (top left, top right, bottom left, bottom right...). The points on normal images is assumed as the 4 edge points defined using width and height of the image.

## For q1 (Affine Rectification)
### Math Involved
1. We need to estimate a transformation upto an affine transform, so we need to solve for 2 D.O.F, as 6 D.O.F freedom would be taken care of by affine transformation
2. Under a projective transformation lines remain lines, so in the input image we can find a line at infinity as the line passing through points through intersection of two lines in input images that are parallel in affine rectified image. In affine rectified image that line is [0, 0, 1].
3. If the line at infinity in input image in (l1, l2, l3) then the transformation is given as

      $$ H=\left[\begin{array}{ccc}
        1 & 0 & 0\\
        0 & 1 & 0\\
        l_1 & l_2 & l_3
        \end{array}\right]$$

### Code Structure
#### Summary
1. q1.py corresponds to carrying out affine rectification. q1.py takes an argument named image_name, which is the image name in ./data/q1/image_name.jpg. Do not include extensions like (.jpg) while parsing arguments, default extends .jpg and programs expect images to be in ./data/q1/

    example  ```$ python3 q1.py --image_name=tiles4```
2. The program asks you if you want to select annotations or use the stored annotations. If you are selecting, you must select two pairs of parallel lines. Each consecutive points define a line, and each consecutive lines are expected to be parallel in affine rectified image.
3. From 8 $R^2$, points they are converted to $P^3$ by appending 1 to each row of the list. 
4. To find the lines cross product of each corresponding points is used, thereby resulting 4 lines.

    $l_1 = a \times b $  where a, b are consecutive points and $l_1$ is the resulting line in $P^3$. The line is normalized.
5. From 4 lines obtained, two intersection points of two consecutive lines are calculated as cross product $x_1 = l_1 \times l_2$, and normalized
6. From the two points, another line is calculated as cross product which will be the line at infinity for input image $(l_1, l_2, l_3) = normalized(x_1 \times x_2) $. These values are used for calculating H.
7. MyWarp function given in utils is used to warp the image with this H.
8. ```find_affine_rectification_matrix(line_points)``` function takes 8 points and returns the transformation matrix. 
9. All results are saved namely annotated input image, output warped image, test lines annotated on warped image, and json file storing annotated pixel coordinates, and transformation calculated. All the results are stored under corresponding folders under ```./output/q1/```
    


## For q2 (Similarity Rectification)
### Math Involved
1. We need to estimate a transformation upto a similarity transform, so we need to solve for 4 D.O.F, as 4 D.O.F freedom would be taken care of by similarity transformation
2. Angle between two lines is given as         
    $\cos(\theta)=l^TC^*_\infty m=0$, if l and m are perpendicular

    So if we annotate lines in input image that are supposed to be perpendicular in the 3d space, we can find equation for $C^{*'}_\infty$.
    
    $l^{'T}C^{*'}_\infty m^{'} = cos(90) = 0$

    $C^{*'}_\infty$ has 5 parameters, every pair of perpendicular lines gives 1 constraint, so we need 5 pairs. If the image is already affine rectified 3 of them are so, we need only 2 pairs to solve for 2 parameters of 
    $C^{*'}_\infty$.

    After finding $C^{*'}_\infty$    we have $C^{*}_\infty=H*C^{*'}_\infty*H^T$. Taking SVD and finding the singular values we can express

    $$C^{*'}_\infty=U\left[\begin{array}{ccc}
        1 & 0 & 0\\
        0 & 1 & 0\\
        0 & 0 & 0
        \end{array}\right]U^T$$

    $$C^{*'}_\infty=HU\left[\begin{array}{ccc}
        \sigma_1 & 0 & 0\\
        0 & \sigma_2 & 0\\
        0 & 0 & 0
        \end{array}\right]U^TH^T$$

    $$H=\left[\begin{array}{ccc}
        1 /\sqrt\sigma_1 & 0 & 0\\
        0 & 1/\sqrt\sigma_2 & 0\\
        0 & 0 & 1\
        \end{array}\right]U^T$$
        
     H satisfies with equation, where $\sigma_1, \sigma_2$ are singular values, and U is the left orthonormal matrix.
3. #### In Program

    1. q2.py takes an argument named --image_name which, takes name of the image (assumes jpg). 
    2. As we are working on affine rectified images, we load the parallel annotations for affine rectification from json file saved from q1.py and calculate the affine rectification transformation, in q2, and warp the image. 
    3. We then annotate or load (if already saved to save time of annotation) 2 sets of lines that should be perpendicular in the rectified image. 
    4. From these annotations, we find 4 lines (2 pairs of perpendicular), 
    $$l = a \times b$$
    5. From 2 pairs of lines, we set linear equations of form $Ah=0$, which we can solve by SVD
    $$A_i=[l_{i1}m_{i1},l_{i2}m_{i1}+l_{i1}m_{i2},l_{i2}m_{i2}]$$
    $$h=[a, b/2,c]$$
    6. From 2 pairs we get a matrix of size 2*, and h has 3 elements (1 scale). 
    7. From h (after dividing by $h_3$), we can build 
    
    $$C^{*'}_\infty= \left[\begin{array}{ccc}
                a & b/2 & 0\\
                b/2 & c & 0\\
                0 & 0 & 0
    \end{array}\right]$$
    
    8. Then we find $\sigma_1, \sigma_2, U$ from SVD of $C^{*'}_\infty$. And 
        
        $$H=\left[\begin{array}{ccc}
                1 /\sqrt\sigma_1 & 0 & 0\\
                0 & 1/\sqrt\sigma_2 & 0\\
                0 & 0 & 1\
                \end{array}\right]U^T$$

    9. Then MyWarp is used to warp input image with H
    10.  All results are saved namely perpendicular lines annotated input image, output warped image, test lines annotated on warped image, and json file storing annotated pixel coordinates, and transformation calculated. All the results are stored under corresponding folders under ```./output/q2/```

## For q3 (Homography)
### Math Involved
1. We need to estimate a homography of 8 D.O.F, using point correspondences in two images. We need 4 pairs of point annotations in normal and perspective image
2. We need to setup $Ah=0$, where $h$ is the 9*1, homography matrix. $H$ has 8 DOF, so we set $||h||=1$ to avoid the trivial solution.
3. Each point correspondence yields two constraints, so we 4 pairs to estimate the 8 parameters. We have $x^T=[x_1, x_2, x_3], x^{'}=[x^{'}, y^{'}, w^{'}]$

    $$A=\left[\begin{array}{ccc}
        0 & -w^{'}*x^T & y^{'}x^T\\
        w^{'}x^T & 0 & -x^{'}x^T\\
        \end{array}\right]$$   

4. From 4 correspondences we get A matrix of size 8*9, and we can solve h as the last row of right orthonormal vector. 
5. Then warp normal-image to an image with size = perspective-image.shape. We then make all points defined by a polygon with coordinates as 4 coordinates of the normal-image, and fill it with (0, 0, 0) RGB color.
6. We bitwise add warped normal image and black polygon filled perspective image.
7. #### In Code
    1. q3.py, takes 2 arguments 
    
        a. first_image_name=name of the normal image (include extension .jpg). 
        
        b. second_image_name=name of the perspective image on which normal image should be warped on to (include extension like .jpg)
        
        Program expects these files to be in ```./data/q3/``` folder
    2. The corresponding points in perspective is selected/loaded(if already annotated) annotated in the following order (top-left, top-right, bottom-left & bottom-right). The corresponding points in normal image are end coordinates ((0, 0), (0, width-1), (height-1, 0), (width-1, height-1))
    3. H is found following equations discussed above. Normal image is warped into an image of size perspective image.
    4. We then make all points defined by a polygon with coordinates as 4 coordinates of the normal-image, and fill it with (0, 0, 0) RGB color.
    5. We bitwise add warped normal image and black polygon filled perspective image.
    6. Then code then saves all results like annotated perspective image, normal and perspective image annotations, and warped and overlaid image, and info of homography and corresponding points.

## For q5 (Homography with mulitple normal images)
### Math Involved
1. Implementation Similar to Question 3.
2. If we have n normal points, we annotated 4*n corresponding points
3. Then we find n Homography Matrices, and warp each image separately
4. Fill the perspective Image with black pixels inside points that will be covered by normal imaged, and add using bitwise or