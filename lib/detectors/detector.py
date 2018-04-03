import itertools
import numpy as np
import lib.structures.basic_geometry as shapes
import lib.utils.geometry as geom

class QuadDetector:
    """Base class for Quad detectors"""
    def detect_quad(self, img):
        raise NotImplemented("A detector must implement this")

    @staticmethod
    def _find_corners(lines):
        pairs = [pair for pair in itertools.combinations(lines, 2)]
        distances = []
        for pair in pairs:
            l1 = shapes.create_line_segment_from_np(pair[0])
            l2 = shapes.create_line_segment_from_np(pair[1])
            distances.append(geom.line_line_distance(l1.a, l1.b, l2.a, l2.b))

        try:
            pairs.pop(np.argmax(distances))
            idx = np.in1d(pairs[0], pairs[1])
            base = shapes.create_line_segment_from_np(np.concatenate(pairs[0])[idx])
            line_list = [shapes.create_line_segment_from_np(line) for line in lines.tolist()]
            line_list.remove(base)
        except ValueError:
            return None

        inner = []
        outer = []
        for line in line_list:
            A = ((line.b.y - line.a.y, line.a.x - line.b.x),
                 (base.b.y - base.a.y, base.a.x - base.b.x))
            B = (line.a.x * line.b.y - line.b.x * line.a.y,
                 base.a.x * base.b.y - base.b.x * base.a.y)
            A = np.stack(A)
            try:
                intersect = np.linalg.solve(A,B)
            except np.linalg.LinAlgError:
                return None

            dist = [geom.distance(intersect, point) for point in line.get_endpoints()]
            inner.append(shapes.Point2D(intersect[0], intersect[1]))
            outer.append(line[np.argmax(dist)])

        return [outer[0], inner[0], inner[1], outer[1]]
