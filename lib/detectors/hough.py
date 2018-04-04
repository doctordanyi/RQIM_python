import lib.detectors.detector as detector
import cv2
import numpy as np
import lib.graphics.renderer as rend
import lib.structures.basic_geometry as geom
import lib.utils.geometry as bg
import lib.structures.quad as quad
from lib.detectors.detector import BaseNotFound, IntersectionNotFound


class NoLinesDetected(Exception):
    pass


class HoughQuadDetector(detector.QuadDetector):
    def __init__(self):
        self._working_img = None
        self._skeleton = None
        self._quad_box_size = None

    def _init_iteration(self, img):
        self._orig_img = img
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self._working_img = cv2.threshold(img, thresh=127, maxval=255, type=cv2.THRESH_BINARY_INV)[1]
        self._set_bounding_box_size()
        self._skeletonize()

    def _set_bounding_box_size(self):
        contours = cv2.findContours(self._working_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        min_rect = cv2.minAreaRect(contours[1][0])
        box = cv2.boxPoints(min_rect)
        dist = [bg.distance(box[0], pt) for pt in box[1:]]
        dist.sort()
        self._quad_box_size = tuple(dist[0:-1])

    def _skeletonize(self):
        self._skeleton = np.zeros(self._working_img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        done = False
        img = np.copy(self._working_img)
        while not done:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            self._skeleton = cv2.bitwise_or(self._skeleton, temp)
            img = np.copy(eroded)
            done = cv2.countNonZero(img) == 0

    def _scale_to_quad_space(self, corners):
        img_size = self._working_img.shape
        return [corner.scale_inhomogen(1/img_size[0], 1/img_size[1]).translate((-0.5, -0.5)) for corner in corners]

    def detect_quad(self, img):
        self._init_iteration(img)

        try:
            segments = self._find_segments()
        except NoLinesDetected:
            return None

        line_segments = []
        for seg in segments:
            line_segments.append(np.concatenate(seg.get_endpoints()))

        if len(line_segments) < 3:
            print("Less than three")
            return None

        line_segments = np.stack(line_segments)
        try:
            corners = self._find_corners(line_segments)
        except (BaseNotFound, IntersectionNotFound):
            return None

        return quad.Quad(self._scale_to_quad_space(corners))


class ClassicHoughDetector(HoughQuadDetector):
    def __init__(self):
        HoughQuadDetector.__init__(self)

    def _get_threshold(self):
        min_dist = np.min(self._quad_box_size)
        if min_dist < 15:
            return 5
        return int(min_dist / 3)

    def _find_lines(self):
        thresh = self._get_threshold()

        lines = cv2.HoughLines(self._skeleton, 1, np.pi / 180, thresh)
        if lines is None or len(lines) < 3:
            raise NoLinesDetected
        lines = [line[0] for line in lines]
        lines = np.stack(lines)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0)
        compactness, labels, centers = cv2.kmeans(lines, 3, None, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)

        return centers

    def _find_segments(self):
        lines = self._find_lines()

        segments = []
        for rho, theta in lines:
            cos = np.cos(theta)
            sin = np.sin(theta)
            if np.isclose(sin, 0):
                line_points = ((y, int(((rho - y * sin) / cos))) for y in range(self._working_img.shape[0]))
            else:
                line_points = ((int((rho - x * cos) / sin), x) for x in range(self._working_img.shape[1]))

            try:
                max_line_gap = int(min(self._quad_box_size) / 2)
                line_points = (pt for pt in line_points if is_valid_index(pt, self._working_img.shape))
                segment_points = [(pt[1], pt[0]) for pt in line_points if self._working_img[pt]]
                segments.append(geom.LineSegment2D(segment_points[0], segment_points[-1], 0))
            except IndexError:
                raise NoLinesDetected

        return segments


class ProbabilisticHoughDetector(HoughQuadDetector):
    def __init__(self):
        HoughQuadDetector.__init__(self)

    def _get_threshold(self):
        min_dist = np.min(self._quad_box_size)
        return int(max((5, min_dist / 6)))

    def _merge_line_segments(self, segments, dir_vec):
        points = [seg.a for seg in segments]
        points.extend([seg.b for seg in segments])

        base = points.pop()
        line_vectors = [point - base for point in points]
        skalar_produkts = [np.dot(dir_vec, line_vec) for line_vec in line_vectors]

        min_value = min(skalar_produkts)
        max_value = max(skalar_produkts)
        min_idx = np.argmin(skalar_produkts)
        max_idx = np.argmax(skalar_produkts)

        if min_value < 0 and max_value < 0:
            return geom.LineSegment2D(points[min_idx], base, 0)

        if min_value > 0 and max_value > 0:
            return geom.LineSegment2D(points[max_idx], base, 0)

        return geom.LineSegment2D(points[min_idx], points[max_idx], 0)

    def _merge_segments(self, segments):
        dir_vectors = [seg.get_dir_vector() for seg in segments]
        dir_vectors = np.array(dir_vectors, dtype=np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0)
        compactness, labels, centers = cv2.kmeans(dir_vectors, 3, None, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)

        segment_clusters = [[],[],[]]
        for label, segment in zip(labels.tolist(), segments):
            segment_clusters[label[0]].append(segment)

        merged_segments = []
        for dir_vec, lines in zip(centers, segment_clusters):
            if len(lines) > 1:
                merged_segments.append(self._merge_line_segments(lines, dir_vec))
            else:
                merged_segments.append(lines[0])

        return merged_segments

    def _find_segments(self):
        thresh = self._get_threshold()
        min_line_length = min(self._quad_box_size) / 4

        lines = cv2.HoughLinesP(self._working_img, 1, np.pi / 180, thresh, minLineLength=min_line_length, maxLineGap=min_line_length/2)
        if lines is None or len(lines) < 3:
            raise NoLinesDetected

        line_segments = [geom.create_line_segment_from_np(seg[0]) for seg in lines]
        line_segments = self._merge_segments(line_segments)

        # if __debug__:
        #     img = np.copy(self._orig_img)
        #     print(len(lines))
        #
        #     for line in line_segments:
        #         color = (random.uniform(50,255), random.uniform(50,255), random.uniform(50,255))
        #         # print("(" + str(x1) + "," + str(y1) + "),(" + str(x2) + "," + str(y2) + ")")
        #         cv2.line(img, line.a, line.b, color, 2)
        #
        #     cv2.imshow('houghlinesP', img)
        #     cv2.imshow('skeleton', self._skeleton)
        #     cv2.waitKey()
        #
        # print("Segments found: " + str(len(line_segments)))
        return line_segments


def is_valid_index(idx, shape):
    for i, lim in zip(idx, shape):
        if i < 0:
            return False
        if i >= lim:
            return False
    return True


def test():
    renderer = rend.Renderer(channels=3)
    detector = ProbabilisticHoughDetector()
    orig = quad.create_random(0.3, 0.2, 0.7, 0.2, 2)
    img = renderer.render(orig)
    detected = detector.detect_quad(img)
    img_rec = renderer.render(detected)
    img_det = cv2.bitwise_not(cv2.cvtColor(img_rec, cv2.COLOR_BGR2GRAY))
    img = cv2.bitwise_not(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    diff = img - img_det
    cv2.imshow("original", img)
    cv2.imshow("detected", img_det)
    cv2.imshow("diff", diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test()
