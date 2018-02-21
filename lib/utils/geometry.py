import math
import numpy as np


def distance(point_a, point_b):
    return np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def create_2d_rotation(rads):
    """Creates a 2D rotation matrix as a numpy array"""
    return np.array([[math.cos(rads), -math.sin(rads)],
                     [math.sin(rads), math.cos(rads)]])


def rotate(point, theta):
    """Rotate a point with theta radians"""
    rot = create_2d_rotation(theta)
    return np.dot(rot, point)


def _on_segment(p, q, r):
    if (max(p.x, r.x) >= q.x >= min(p.x, r.x) and
            max(p.y, r.y) >= q.y >= min(p.y, r.y)):
        return True

    return False


def _orientation(p, q, r):
    """Return the orientation of the triangle made by 3 points"""
    eps = 0.00000001

    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if math.isclose(val, 0, abs_tol=eps):
        return 0

    if val > 0:
        return 1

    return -1


def line_segment_intersect(p1, p2, q1, q2):
    """Check if two line segments intersect. Return true/false"""
    # Get the orientations of the 4 possible triangles
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    # intersecting line segment
    if (o1 != o2) and (o3 != o4):
        return True

    # collinear cases
    # p1, q1 and p2 are collinear and p2 lies on segment p1q1
    if o1 == 0 and _on_segment(p1, p2, q1):
        return True

    # p1, q1 and p2 are collinear and q2 lies on segment p1q1
    if o2 == 0 and _on_segment(p1, q2, q1):
        return True

    # p2, q2 and p1 are Collinear and p1 lies on segment p2q2
    if o3 == 0 and _on_segment(p2, p1, q2):
        return True

    # p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if o4 == 0 and _on_segment(p2, q1, q2):
        return True

    return False  # Doesn't fall in any of the above cases


def line_point_distance(p1, p2, r):
    try:
        u = (r.x - p1.x)*(p2.x - p1.x) + (r.y - p1.y)*(p2.y - p1.y) / (distance(p1, p2)**2)
        if u < 0:
            return distance(p1, r)
        if u > 1:
            return distance(p2, r)
        p = (p1.x + u*(p2.x - p1.x), p1.y + u*(p2.y - p1.y))
        return distance(p, r)
    except ZeroDivisionError:
        return distance(p1, r)


def line_line_distance(p1, p2, q1, q2):
    return min(line_point_distance(p1, p2, q1),
               line_point_distance(p1, p2, q2))
