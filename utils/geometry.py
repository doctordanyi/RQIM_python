import math
import numpy as np


def distance(point_a, point_b):
    return np.hypot(point_a.x - point_b.x, point_a.y - point_b.y)


def create_2d_rotation(rads):
    """Creates a 2D rotation matrix as a numpy array"""
    return np.array([[math.cos(rads), -math.sin(rads)],
                     [math.sin(rads), math.cos(rads)]])


def rotate(point, theta):
    """Rotate a point with theta radians"""
    rot = create_2d_rotation(theta)
    return np.dot(rot, point)

