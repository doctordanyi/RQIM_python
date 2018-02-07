import numpy as np
import math as m
import utils.math_common as mc


class Quad:

    __round_digits = 10

    def __init__(self, corners):
        self.corners = tuple(corners)

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
            rot = mc.create_2d_rotation(orientation)
            corners = [np.dot(rot, x) for x in corners]

        corners = [np.around(x, cls.__round_digits) for x in corners]
        return cls(corners)

    @classmethod
    def create_from_corners(cls, end1, inner1, inner2, end2):
        return cls([end1, inner1, inner2, end2])

    @classmethod
    def create_random(cls):
        return cls(end1=(0, 1), end2=(0, 0), inner1=(1, 0), inner2=(1, 1))


def test():
    quad = Quad.create_from_params(4, m.pi / 4, m.pi / 4, m.sqrt(2) / 4, m.sqrt(2) / 4, 0)
    print(quad.corners)


if __name__ == "__main__":
    test()
