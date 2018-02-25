import cv2
import numpy as np
import lib.utils.geometry as geom


class QuadDetector:
    """Abstract class for Quad detector interface definition"""
    def detect_quad(self, img):
        raise NotImplemented


class LSDQuadDetector(QuadDetector):
    def __init__(self):
        self.lsd = cv2.createLineSegmentDetector()

    def _find_parallel(self, point, other_points):
        distances = [geom.distance(point, -other) for other in other_points]
        return np.argmin(distances)

    def _find_pairs(self, lines):
        dir_vectors = []
        for line in lines:
            dir_vector = line[0, 2:4] - line[0, 0:2]
            dir_vectors.append(dir_vector / np.linalg.norm(dir_vector))

        pairs = []
        for i in range(len(dir_vectors)):
            temp_vectors = dir_vectors.copy()
            # temp_vectors.remove(dir_vec)
            min_idx = self._find_parallel(dir_vectors[i], temp_vectors)
            idx_pair = [min_idx, i]
            idx_pair.sort()
            pairs.append((lines[idx_pair[0]], lines[idx_pair[1]]))

        pairs = np.stack(pairs)
        pairs = np.unique(pairs)
        return pairs

    def detect_quad(self, img):
        lines, widths, prec, nfa = self.lsd.detect(img)
        # Both edges of every line segment is found
        if lines.shape[0] == 6:
            pairs = self._find_pairs(lines)


detector = LSDQuadDetector()
img = cv2.imread("out/quad74.png", 0)
detector.detect_quad(img)

# mod_lines = []
# for line in lines:
#     print(line[0])
#     delta = line[0, 2:4] - line[0, 0:2]
#     delta = (delta / np.linalg.norm(delta)) * 5/2
#     delta = geom.rotate(delta, np.pi / 2)
#     delta = np.concatenate((delta, delta))
#     print(delta)
#     mod_lines.append([line[0] + delta])
#
# mod_lines = np.stack(mod_lines).astype(np.float32)
# blank = np.zeros(img.shape, np.uint8)
# drw_orig = lsd.drawSegments(blank, lines)
# drw_mod = lsd.drawSegments(blank, mod_lines)
# cv2.imshow("detected orig", drw_orig)
# cv2.imshow("detected mod", drw_mod)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

