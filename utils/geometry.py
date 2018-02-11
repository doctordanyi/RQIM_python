import math
import numpy as np


def distance(point_a, point_b):
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def create_2d_rotation(rads):
    """Creates a 2D rotation matrix as a numpy array"""
    return np.array([[math.cos(rads), -math.sin(rads)],
                     [math.sin(rads), math.cos(rads)]])


def rotate(point, theta):
    """Rotate a point with theta radians"""
    rot = create_2d_rotation(theta)
    return np.dot(rot, point)

