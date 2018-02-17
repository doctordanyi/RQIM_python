from collections import namedtuple
from utils.geometry import rotate


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

