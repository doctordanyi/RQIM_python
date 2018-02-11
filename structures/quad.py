import numpy as np
import math as m
import utils.geometry as geom


class Quad:

    __round_digits = 10

    def __init__(self, corners):
        self.corners = tuple(corners)

    def __str__(self):
        str = "Quad ("
        for corner in self.corners:
            str = str + "(" + corner[0] + "," + corner[1] + "),"
        str = str + ")"
        return str

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
    def create_random(cls):
        return cls(((0, 1), (0, 0), (1, 0), (1, 1)))

    def rotate(self, orientation):
        """Rotates the current quad with [orientation] radians. Returns a new instance"""
        corners = [geom.rotate(x, orientation) for x in self.corners]
        corners = [np.around(x, self.__round_digits) for x in corners]
        return Quad(corners)

    def get_base_length(self):
        return geom.distance(self.corners[1], self.corners[2])

    def get_area(self):
        a = (self.corners[0][0] - self.corners[2][0], self.corners[0][1] - self.corners[2][1])
        b = (self.corners[1][0] - self.corners[3][0], self.corners[1][1] - self.corners[3][1])
        return m.sqrt(0.5 * m.fabs((a[0] - b[1]) * (a[1] - b[0])))


def test():
    quad = Quad.create_from_params(4, m.pi / 4, m.pi / 4, m.sqrt(2) / 4, m.sqrt(2) / 4, 0)
    print(quad.corners)


if __name__ == "__main__":
    test()
