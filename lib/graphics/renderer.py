import cv2
import numpy as np
from lib.structures.quad import Quad
from collections import namedtuple


class Renderer:
    Size = namedtuple("Size", "height width channels")

    def __init__(self, height=640, width=640, channels=3):
        self.size = self.Size(height, width, channels)

    def draw_line(self, img, end1, end2, width):
        cv2.line(img=img, pt1=end1, pt2=end2, color=(0, 0, 0), lineType=cv2.LINE_AA, thickness=width)

    def _cvt_to_img_coords(self, point):
        return (int((point.x + 0.5) * self.size.width),
                int((point.y + 0.5) * self.size.height))

    def _render_quad(self, quad, img):
        thickness_multiplier = 0.01
        img_points = [self._cvt_to_img_coords(pt) for pt in quad.corners]
        width = min(255, 1 + int(np.around(thickness_multiplier * self.size.width * quad.get_base_length(), 0)))

        for i in range(3):
            self.draw_line(img, img_points[i], img_points[i+1], width)

    def render(self, obj):
        if type(obj) is Quad:
            img = np.zeros(self.size, np.uint8)
            img[:] = (255, 255, 255)
            self._render_quad(obj, img)
            return img
        else:
            raise TypeError("Rendering is not defined for " + str(type(obj)))
