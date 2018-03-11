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
        # order te corners based on the x coordinate of inner points
        if corners[1][0] > corners[2][0]:
            corners[1], corners[2] = corners[2], corners[1]
            corners[0], corners[3] = corners[3], corners[0]
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

    def get_b_angle(self):
        a = self.corners[0]-self.corners[1]
        b = self.corners[2] - self.corners[1]
        return np.arccos(np.dot(a,b) / (np.hypot(a.x, a.y)*np.hypot(b.x, b.y)))

    def get_c_angle(self):
        a = self.corners[3]-self.corners[2]
        b = self.corners[1] - self.corners[2]
        return np.arccos(np.dot(a, b) / (np.hypot(a.x, a.y)*np.hypot(b.x, b.y)))

    def get_b_multiplier(self):
        return geom.distance(self.corners[0], self.corners[1]) / self.get_base_length()

    def get_c_multiplier(self):
        return geom.distance(self.corners[3], self.corners[2]) / self.get_base_length()

    def get_orientation(self):
        base = self.corners[1] - self.corners[2]
        base = base.rotate(np.pi / 2)
        return np.arctan2(base.x, base.y)

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

    def fix_winding(self):
        area2 = (self.corners[1].x - self.corners[0].x) * (self.corners[1].y + self.corners[0].y) + \
                (self.corners[2].x - self.corners[1].x) * (self.corners[2].y + self.corners[1].y) + \
                (self.corners[3].x - self.corners[2].x) * (self.corners[3].y + self.corners[2].y) + \
                (self.corners[0].x - self.corners[3].x) * (self.corners[0].y + self.corners[3].y)

        if area2 > 0:
            self.corners = (self.corners[3], self.corners[2], self.corners[0], self.corners[1])


class QuadEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Quad):
            return dict(type='Quad', corners=o.corners, base_length=o.get_base_length())
        return json.JSONEncoder.default(self, o)


class QuadProvider:
    """Abstract class defining the QuadProvider interface"""
    def get_quads(self):
        raise NotImplemented

    def get_quad_groups(self):
        """Should return an iterable of dictionaries containing the following information
           group_by: name of the parameter used to create groups
           param: the parameter of current group
           quads: the list of quads in the current group"""
        raise NotImplemented


class QuadGenerator(QuadProvider):
    def __init__(self, quads_per_scale = 100, base_lengths = [0.5], ca_range = [0.2, 1], alpha_range = [0.3, 2]):
        self.quads_per_scale = quads_per_scale
        self.scales = list(base_lengths)
        self.ca_min = ca_range[0]
        self.ca_max = ca_range[1]
        self.alpha_min = alpha_range[0]
        self.alpha_max = alpha_range[1]
        self.quads = []

    def generate(self):
        for scale in self.scales:
            i = 0
            quads = []
            while i < self.quads_per_scale:
                q = create_random(scale, 0.2, 1, 0.3, 2)
                if q.is_valid():
                    quads.append(q)
                    i += 1
            self.quads.append(quads)

    def clear_quads(self):
        self.quads.clear()

    def get_all_quads(self):
        return [q for quad_list in self.quads for q in quad_list]

    def save_to_json(self, filename):
        with open(filename, 'w') as json_file:
            ranges = {"ca_min": self.ca_min,
                      "ca_max": self.ca_max,
                      "alpha_min": self.alpha_min,
                      "alpha_max": self.alpha_max}
            header = {"quads_per_scale": self.quads_per_scale,
                      "random_ranges": ranges,
                      "scales": self.scales}
            quads = [{"nominal_base_length": base, "quads": quads} for (base, quads) in zip(self.scales, self.quads)]
            data = {"generation_parameters": header,
                    "data": quads}
            json.dump(data, json_file, cls=QuadEncoder, separators=(',', ':'), indent=4)

    def get_quads(self):
        return self.get_all_quads()

    def get_quad_groups(self):
        for (base, quad_list) in zip(self.scales, self.quads):
            yield dict(group_by="base_length", param=base, quads=quad_list)


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


def get_abs_sum_position_error(orig, other):
    abs_diff = 0
    for i in range(4):
        abs_diff += geom.distance(orig.corners[i], other.corners[i])

    return abs_diff


def get_abs_avg_position_error(orig, other):
    return get_abs_sum_position_error(orig, other) / 4


def get_rel_avg_position_error(orig, other):
    rel_diff_sum = 0
    for i in range(4):
        rel_diff_sum += geom.distance(orig.corners[i], other.corners[i]) / np.linalg.norm(orig.corners[i])

    return rel_diff_sum / 4


def get_abs_sum_angle_error(orig, other):
    return abs(orig.get_b_angle() - other.get_b_angle()) + abs(orig.get_c_angle() - other.get_c_angle())


def get_abs_avg_angle_error(orig, other):
    return get_abs_sum_angle_error(orig, other) / 2


def get_rel_avg_angle_error(orig, other):
    b_angle_err = abs((orig.get_b_angle() - other.get_b_angle()) / orig.get_b_angle())
    c_angle_err = abs((orig.get_c_angle() - other.get_c_angle()) / orig.get_c_angle())
    return (b_angle_err + c_angle_err) / 2


def get_abs_sum_multiplier_error(orig, other):
    return abs(orig.get_b_multiplier() - other.get_b_multiplier()) + \
           abs(orig.get_c_multiplier() - other.get_c_multiplier())


def get_abs_avg_multiplier_error(orig, other):
    return get_abs_sum_multiplier_error(orig, other) / 2


def get_rel_avg_multiplier_error(orig, other):
    b_multiplier_err = abs((orig.get_b_multiplier() - other.get_b_multiplier()) / orig.get_b_multiplier())
    c_multiplier_err = abs((orig.get_c_multiplier() - other.get_c_multiplier()) / orig.get_c_multiplier())
    return (b_multiplier_err + c_multiplier_err) / 2


def get_abs_orientation_error(orig, other):
    return abs(orig.get_orientation() - other.get_orientation())


def get_rel_orientation_error(orig, other):
    return abs((orig.get_orientation() - other.get_orientation()) / orig.get_orientation())


def test():
    quad = create_from_params(4, m.pi / 4, m.pi / 4, m.sqrt(2) / 4, m.sqrt(2) / 4, -np.pi / 2)
    quad.get_orientation()
    print(quad.corners)
    for i in range(20):
        print(create_random(0.5, 0.2, 1, 0, 2))


if __name__ == "__main__":
    test()
