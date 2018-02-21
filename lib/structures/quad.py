import numpy as np
import math as m
import random, json
from lib.structures.basic_geometry import Point2D
import lib.utils.geometry as geom
from collections import namedtuple


class Quad:
    BoundingBox = namedtuple("BoundingBox", "x_min y_min x_max y_max")

    __safety_margin = 36
    __coord_bounds = BoundingBox(-0.5, -0.5, 0.5, 0.5)

    def __init__(self, corners):
        self.corners = tuple([Point2D(p[0], p[1]) for p in corners])

    def __str__(self):
        str_ = "Quad ("
        for corner in self.corners:
            str_.join(str(corner) + ",")
        str_ = str_ + ")"
        return str_

    def rotate(self, orientation):
        """Rotates the current quad with [orientation] radians. Returns a new instance"""
        corners = [x.rotate(orientation) for x in self.corners]
        return Quad(corners)

    def get_base_length(self):
        return geom.distance(self.corners[1], self.corners[2])

    def get_area(self):
        a = (self.corners[0].x - self.corners[2].x, self.corners[0].y - self.corners[2].y)
        b = (self.corners[1].x - self.corners[3].x, self.corners[1].y - self.corners[3].y)
        return m.sqrt(0.5 * m.fabs((a[0] - b[1]) * (a[1] - b[0])))

    def is_valid(self):
        """Performs semantic check on the quad. return true if valid"""
        for corner in self.corners:
            if corner.x < self.__coord_bounds.x_min or \
                    corner.x > self.__coord_bounds.x_max or \
                    corner.y < self.__coord_bounds.y_min or \
                    corner.y > self.__coord_bounds.y_max:
                return False

        if geom.line_segment_intersect(self.corners[0], self.corners[1],
                                       self.corners[2], self.corners[3]):
            return False

        end_segment_dist = geom.line_line_distance(self.corners[0], self.corners[1],
                                                   self.corners[2], self.corners[3])
        if end_segment_dist * 6 < self.get_base_length():
            return False

        end_dist = geom.distance(self.corners[0], self.corners[3])
        if end_dist * 6 < self.get_base_length():
            return False

        return True


class QuadEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Quad):
            return dict(type='Quad', corners=o.corners, base_length=o.get_base_length())
        return json.JSONEncoder.default(self, o)


def create_from_params(base_length, b_angle, c_angle, b_multiplier, c_multiplier, orientation):
    cos = np.cos([b_angle, np.pi - c_angle])
    sin = np.sin([b_angle, np.pi - c_angle])

    corners = [(cos[0] * base_length * b_multiplier - base_length / 2,
                sin[0] * base_length * b_multiplier),
               (-base_length / 2, 0),
               (base_length / 2, 0),
               (cos[1] * base_length * c_multiplier + base_length / 2,
                sin[1] * base_length * c_multiplier)]

    if orientation:
        corners = [geom.rotate(x, orientation) for x in corners]

    return Quad(corners)


def create_random(a, ca_min, ca_max, alpha_min, alpha_max):
    return create_from_params(base_length=a,
                              b_angle=random.uniform(alpha_min, alpha_max),
                              c_angle=random.uniform(alpha_min, alpha_max),
                              b_multiplier=random.uniform(ca_min, ca_max),
                              c_multiplier=random.uniform(ca_min, ca_max),
                              orientation=random.uniform(0, 2 * m.pi))


def create_from_corners(end1, inner1, inner2, end2):
    return Quad([end1, inner1, inner2, end2])


def test():
    quad = create_from_params(4, m.pi / 4, m.pi / 4, m.sqrt(2) / 4, m.sqrt(2) / 4, m.pi / 4)
    print(quad.corners)
    for i in range(20):
        print(create_random(0.5, 0.2, 1, 0, 2))


if __name__ == "__main__":
    test()
