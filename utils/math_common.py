import numpy as np
import math as m


def create_2d_rotation(rads):
    """Creates a 2D rotation matrix as a numpy array"""
    return np.array([[m.cos(rads), -m.sin(rads)],
                     [m.sin(rads), m.cos(rads)]])


def test():
    x = create_2d_rotation(m.pi / 4)
    print(x)


if __name__ == "__main__":
    test()
