import lib.detectors.detector as detector
import cv2
import numpy as np
import lib.graphics.renderer as rend
import lib.structures.basic_geometry as geom
import lib.utils.geometry as bg


class HoughLinesQuadDetector(detector.QuadDetector):
    @staticmethod
    def _get_threshold(gray):
        contours = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        min_rect = cv2.minAreaRect(contours[1][0])
        box = cv2.boxPoints(min_rect)
        dist = [bg.distance(box[0], pt) for pt in box[1:]]
        min_dist = np.min(dist)
        if min_dist < 15:
            return 5

        return int(min_dist / 3)

    def _skeleton(self, img):
        skeleton = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        done = False
        while not done:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            img = np.copy(eroded)
            done = cv2.countNonZero(img) == 0

        return skeleton

    def _find_lines(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, thresh=127, maxval=255, type=cv2.THRESH_BINARY_INV)[1]
        thresh = self._get_threshold(gray)
        skeleton = self._skeleton(gray)

        cv2.imshow("Skeleton", skeleton)
        cv2.waitKey(0)

        lines = cv2.HoughLines(skeleton, 1, np.pi / 180, thresh)
        lines = [line[0] for line in lines]
        lines = np.stack(lines)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0)
        compactness, labels, centers = cv2.kmeans(lines, 3, None, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)

        return centers

    def _find_segments(self, img, lines):
        segments = []
        for rho, theta in lines:
            cos = np.cos(theta)
            sin = np.sin(theta)
            if np.isclose(sin, 0):
                line_points = ((y, int(((rho - y * sin) / cos))) for y in range(img.shape[0]))
            else:
                line_points = ((int((rho - x * cos) / sin), x) for x in range(img.shape[1]))

            line_points = (pt for pt in line_points if is_valid_index(pt, img.shape))
            segment_points = [pt for pt in line_points if img[pt]]
            segments.append(geom.LineSegment2D(segment_points[0], segment_points[-1], 0))

        return segments

    def detect_quad(self, img):
        lines = self._find_lines(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, thresh=127, maxval=255, type=cv2.THRESH_BINARY_INV)[1]
        segments = self._find_segments(gray, lines)

        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        for seg in segments:
            cv2.line(img, (seg.a.y, seg.a.x), (seg.b.y, seg.b.x), (0,255,0), 2)

        cv2.imshow("Lines", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def is_valid_index(idx, shape):
    for i, lim in zip(idx, shape):
        if i < 0:
            return False
        if i >= lim:
            return False
    return True


def test():
    detector = HoughLinesQuadDetector()
    img = cv2.imread("../../experiment/out/quad7.png", 1)
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
