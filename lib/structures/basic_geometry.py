from collections import namedtuple
import lib.utils.geometry as geom
import numpy as np


class Point2D(namedtuple("Point2D", "x y")):
    def __new__(cls, x, y):
        return super(Point2D, cls).__new__(cls, x, y)

    def rotate(self, angle):
        """Rotate the point by [angle] radians around origin"""
        x, y = geom.rotate(self, angle)
        return Point2D(x, y)

    def translate(self, other_point):
        """Translates a 2D point with (x,y)"""
        return Point2D(self.x + other_point[0], self.y + other_point[1])

    def average(self, other):
        """Return the average of this point and other_point"""
        return Point2D((self.x + other.x) / 2, (self.y + other.y) / 2)

    def scale(self, factor):
        return Point2D(self.x * factor, self.y * factor)

    def scale_inhomogen(self, factor_x, factor_y):
        return Point2D(self.x * factor_x, self.y * factor_y)

    def __add__(self, other):
        return self.translate(other)

    def __sub__(self, other):
        return self.translate(-other)

    def __neg__(self):
        return Point2D(-self.x, -self.y)


def create_line_segment_from_np(line, width=None):
    if width:
        return LineSegment2D(line[0:2], line[2:4], width)

    return LineSegment2D(line[0:2], line[2:4], 0)


class LineSegment2D(namedtuple("LineSegment2D", "a b width")):
    def __new__(cls, a, b, width):
        return super(LineSegment2D, cls).__new__(cls, Point2D(a[0], a[1]), Point2D(b[0], b[1]), width)

    def get_endpoints(self):
        return self.a, self.b

    def get_length(self):
        return geom.distance(self.a, self.b)

    def get_dir_vector(self):
        dir_vector = self.b - self.a
        norm = np.linalg.norm(dir_vector)
        return Point2D(dir_vector[0] / norm, dir_vector[1] / norm)

    def get_slope_angle(self):
        direction_vec = self.get_dir_vector()
        return np.arctan2(direction_vec[1], direction_vec[0])

    def get_norm_vector(self):
        direction = self.get_dir_vector()
        return direction.rotate(np.pi / 2)


def test_slope_angle():
    line = LineSegment2D((0,0), (-1,0), 0)
    print(line.get_slope_angle())


if __name__ == "__main__":
    test_slope_angle()
