import cv2
import numpy as np
import lib.utils.geometry as geom
import lib.structures.basic_geometry as shapes
import lib.structures.quad as quad
import itertools
import lib.graphics.renderer as rend
from lib.detectors.detector import QuadDetector
from lib.detectors.detector import BaseNotFound, IntersectionNotFound


class LSDQuadDetector(QuadDetector):
    def __init__(self):
        self.lsd = cv2.createLineSegmentDetector()

    def _find_parallel(self, point, other_points):
        distances = [geom.distance(point, -other) for other in other_points]
        return np.argmin(distances)

    def _find_pairs(self, lines):
        dir_vectors = []
        for line in lines:
            dir_vector = line[2:4] - line[0:2]
            dir_vectors.append(dir_vector / np.linalg.norm(dir_vector))

        pairs = []
        for i in range(len(dir_vectors)):
            temp_vectors = dir_vectors.copy()
            min_idx = self._find_parallel(dir_vectors[i], temp_vectors)
            idx_pair = [min_idx, i]
            idx_pair.sort()
            pairs.append((lines[idx_pair[0]], lines[idx_pair[1]]))

        pairs = np.stack(pairs)
        pairs = np.unique(pairs, axis=0)
        return pairs

    def _merge_pairs_long(self, pairs):
        lines = []
        for pair in pairs:
            l1 = shapes.create_line_segment_from_np(pair[0], pair[0][4])
            l2 = shapes.create_line_segment_from_np(pair[1], pair[1][4])
            if l1.get_length() > l2.get_length():
                norm = np.concatenate((l1.get_norm_vector(), l1.get_norm_vector(), [0]))
                norm = norm * (l1.width / 2)
                lines.append(pair[0] + norm)
            else:
                norm = np.concatenate((l2.get_norm_vector(), l2.get_norm_vector(), [0]))
                norm = norm * (l2.width / 2)
                lines.append(pair[1] + norm)

        return np.stack(lines)

    def _scale_to_quad_space(self, corners):
        img_size = self.working_img.shape
        return [corner.scale_inhomogen(1/img_size[0], 1/img_size[1]).translate((-0.5, -0.5)) for corner in corners]

    def detect_quad(self, img):
        self.working_img = img
        try:
            lines, widths, prec, nfa = self.lsd.detect(img)
            lines = [(np.concatenate((lines[i][0], widths[i]))) for i in range(lines.shape[0])]
            lines = np.stack(lines)
        except AttributeError:
            return None

        # Both edges of every line segment is found
        if lines.shape[0] == 6:
            pairs = self._find_pairs(lines)
            lines_merged = self._merge_pairs_long(pairs)
        elif lines.shape[0] == 3:
            lines_merged = lines
        else:
            return None

        try:
            corners = self._find_corners(lines_merged)
        except (BaseNotFound, IntersectionNotFound) as e:
            return None

        if corners:
            return quad.Quad(self._scale_to_quad_space(corners))
        return None


def test():
    detector = LSDQuadDetector()
    img = cv2.imread("out/quad7.png", 0)
    detected = detector.detect_quad(img)
    renderer = rend.Renderer()
    img_rec = renderer.render(detected)
    img_det = cv2.cvtColor(img_rec, cv2.COLOR_BGR2GRAY)
    diff = img - img_det
    cv2.imshow("original", img)
    cv2.imshow("detected", img_det)
    cv2.imshow("diff", diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test()
