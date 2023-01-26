import numpy as np
from utils import normalize, normalize_projective


def find_affine_rectification_matrix(line_points):
    line_equations = []
    for i in range(len(line_points)//2):
        line_equation = np.cross(np.asarray([line_points[2*i][0], 
                                             line_points[2*i][1],
                                             1]),
                                np.asarray([line_points[2*i+1][0], 
                                            line_points[2*i+1][1], 
                                            1]))
        line_equations.append(normalize_projective(line_equation))
    assert(len(line_equations) == 4)

    intersecting_points = []
    # finding line intersections and initial cosines
    for i in range(len(line_equations)//2):
        intersecting_point = np.cross(line_equations[2*i], line_equations[2*i+1])
        intersecting_points.append(normalize_projective(intersecting_point))

    line_infinity = np.cross(intersecting_points[0], intersecting_points[1])
    line_infinity = normalize(line_infinity)
    H = np.asarray([[1, 0, 0],
                    [0, 1, 0],
                    line_infinity])
    return H

if __name__ == '__main__':
    pass