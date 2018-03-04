import cv2
import numpy as np
import lib.utils.geometry as geom
import lib.structures.basic_geometry as shapes
import lib.structures.quad as quad
import itertools
import lib.graphics.renderer as rend


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

    def _merge_pairs(self, pairs):
        lines = []
        for pair in pairs:
            temp = np.copy(pair[1, 2:])
            pair[1, 2:] = pair[1, 0:2]
            pair[1, 0:2] = temp
            lines.append([np.average(pair, axis=0)])

        return np.stack(lines)

    def _merge_pairs_long(self, pairs):
        lines = []
        for pair in pairs:
            l1 = shapes.create_line_segment_from_np(pair[0])
            l2 = shapes.create_line_segment_from_np(pair[1])
            if l1.get_length() > l2.get_length():
                lines.append(pair[0])
            else:
                lines.append(pair[1])

        return np.stack(lines)


    def _merge_endpoints(self, lines):
        pairs = [pair for pair in itertools.combinations(lines, 2)]
        distances = []
        for pair in pairs:
            l1 = shapes.create_line_segment_from_np(pair[0])
            l2 = shapes.create_line_segment_from_np(pair[1])
            distances.append(geom.line_line_distance(l1.a, l1.b, l2.a, l2.b))

        pairs.pop(np.argmax(distances))
        idx = np.in1d(pairs[0], pairs[1])
        base = shapes.create_line_segment_from_np(np.concatenate(pairs[0])[idx])
        line_list = [shapes.create_line_segment_from_np(line) for line in lines.tolist()]
        line_list.remove(base)
        np.linalg.solve()

        # all sides are identified

        outer = []
        inner = []
        for line in line_list:
            distances = []
            for end_point in line:
                distances.append((geom.distance(base.a, end_point), geom.distance(base.b, end_point)))
            distances = np.array(distances)
            ind = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
            inner.append(base[ind[1]].average(line[ind[0]]))
            outer.append(line[1-ind[0]])

        return [outer[0].scale(1/600).translate((-0.5, -0.5)),
                inner[0].scale(1/600).translate((-0.5, -0.5)),
                inner[1].scale(1/600).translate((-0.5, -0.5)),
                outer[1].scale(1/600).translate((-0.5, -0.5))]

    def detect_quad(self, img):
        lines, widths, prec, nfa = self.lsd.detect(img)
        lines = [(np.concatenate((lines[i][0], widths[i]))) for i in range(lines.shape[0])]
        lines = np.stack(lines)

        # Both edges of every line segment is found
        if lines.shape[0] == 6:
            pairs = self._find_pairs(lines)
            lines_merged = self._merge_pairs_long(pairs)
            corners = self._merge_endpoints(lines_merged)
            return quad.Quad(corners)


def test():
    detector = LSDQuadDetector()
    img = cv2.imread("out/quad4.png", 0)
    detected = detector.detect_quad(img)
    renderer = rend.Renderer()
    img_rec = renderer.render(detected)
    cv2.imshow("original", img)
    cv2.imshow("detected", img_rec)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test()
