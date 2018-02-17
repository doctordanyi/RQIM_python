import numpy as np
import math as m
from structures.point import Point2D
import utils.geometry as geom
import random


class Quad:

    __round_digits = 10

    def __init__(self, corners):
        self.corners = tuple([Point2D(p[0], p[1]) for p in corners])

    def __str__(self):
        str_ = "Quad ("
        for corner in self.corners:
            str_ = str_ + str(corner) + ","
        str_ = str_ + ")"
        return str_

    @classmethod
    def create_from_params(cls, base_length, b_angle, c_angle,
                           b_multiplier, c_multiplier, orientation):
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

        corners = [np.around(x, cls.__round_digits) for x in corners]
        return cls(corners)

    @classmethod
    def create_from_corners(cls, end1, inner1, inner2, end2):
        return cls([end1, inner1, inner2, end2])

    @classmethod
    def create_random(cls, a, ca_min, ca_max, alpha_min, alpha_max):
        return cls.create_from_params(base_length=a,
                                      b_angle=random.uniform(alpha_min, alpha_max),
                                      c_angle=random.uniform(alpha_min, alpha_max),
                                      b_multiplier=random.uniform(ca_min, ca_max),
                                      c_multiplier=random.uniform(ca_min, ca_max),
                                      orientation=random.uniform(0, 2 * m.pi))

    def rotate(self, orientation):
        """Rotates the current quad with [orientation] radians. Returns a new instance"""
        corners = [geom.rotate(x, orientation) for x in self.corners]
        corners = [np.around(x, self.__round_digits) for x in corners]
        return Quad(corners)

    def get_base_length(self):
        return geom.distance(self.corners[1], self.corners[2])

    def get_area(self):
        a = (self.corners[0].x - self.corners[2].x, self.corners[0].y - self.corners[2].y)
        b = (self.corners[1].x - self.corners[3].x, self.corners[1].y - self.corners[3].y)
        return m.sqrt(0.5 * m.fabs((a[0] - b[1]) * (a[1] - b[0])))


def test():
    quad = Quad.create_from_params(4, m.pi / 4, m.pi / 4, m.sqrt(2) / 4, m.sqrt(2) / 4, m.pi / 4)
    print(quad.corners)
    for i in range(20):
        print(Quad.create_random(0.5, 0.2, 1, 0, 2))


if __name__ == "__main__":
    test()
