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
        u = ((r.x - p1.x)*(p2.x - p1.x) + (r.y - p1.y)*(p2.y - p1.y)) / (distance(p1, p2)**2)
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


def createLineIterator(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
    """
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
    elif P1Y == P2Y: #horizontal line segment
        itbuffer[:,1] = P1Y
        if negX:
            itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32)/dY.astype(np.float32)
            if negY:
                itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
            itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32)/dX.astype(np.float32)
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

    return itbuffer
