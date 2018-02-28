from collections import namedtuple
from lib.utils.geometry import rotate


class Point2D(namedtuple("Point2D", "x y")):
    def __new__(cls, x, y):
        return super(Point2D, cls).__new__(cls, x, y)

    def rotate(self, angle):
        """Rotate the point by [angle] radians around origin"""
        x, y = rotate(self, angle)
        return Point2D(x, y)

    def translate(self, other_point):
        """Translates a 2D point with (x,y)"""
        return Point2D(self.x + other_point[0], self.y + other_point[1])


def create_line_segment_from_np(line):
    return LineSegment2D(line[0,0:2], line[0,2:4])


class LineSegment2D(namedtuple("LineSegment2D", "a b")):
    def __new__(cls, a, b):
        return super(LineSegment2D, cls).__new__(cls, Point2D(a[0], a[1]), Point2D(b[0], b[1]))
