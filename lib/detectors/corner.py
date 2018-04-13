from lib.detectors.detector import QuadDetector, BaseNotFound
import lib.graphics.renderer as rend
import lib.structures.quad as quad
import lib.utils.geometry as geom
import lib.structures.basic_geometry as bg
import cv2
import numpy as np
import itertools, random


class CornerQuadDetector(QuadDetector):
    def __init__(self):
        self._working_img = None

    def _init_iteration(self, img):
        self._orig_img = img
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self._working_img = img

    def _identify_points(self, centroids):
        binary = cv2.threshold(self._working_img, thresh=127, maxval=255, type=cv2.THRESH_BINARY_INV)[1]

        pointpairs = itertools.combinations(centroids.tolist(), 2)
        lines = []
        for pointpair in pointpairs:
            line_points = geom.createLineIterator(np.array(pointpair[0]), np.array(pointpair[1]), binary)
            hit_miss_ratio = np.count_nonzero(line_points[:,2]) / len(line_points)
            lines.append((hit_miss_ratio, pointpair))

        lines.sort(key=lambda line: line[0])
        lines = lines[-3:]
        lines = [line[1:][0] for line in lines]

        line_points = [point for line in lines for point in line]
        outer = []
        inner = []
        for centroid in centroids.tolist():
            point_occurance_count = line_points.count(centroid)
            if point_occurance_count == 1:
                outer.append(centroid)
            elif point_occurance_count == 2:
                inner.append(centroid)
            else:
                raise BaseNotFound

        return inner, outer

    def _scale_to_quad_space(self, corners):
        img_size = self._working_img.shape
        return [corner.scale_inhomogen(1/img_size[0], 1/img_size[1]).translate((-0.5, -0.5)) for corner in corners]

    def detect_quad(self, img):
        self._init_iteration(img)

        gray = np.float32(self._working_img)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        # dst += -1 * dst.min()
        # dst *= 255.0 / dst.max()
        # dst = np.array(dst, dtype=np.uint8)

        # result is dilated for marking the corners, not important
        centroids = None
        for i in range(10):
            binary_corners = cv2.threshold(dst, thresh=0.01 * dst.max(), maxval=255, type=cv2.THRESH_BINARY)[1]
            binary_corners = np.uint8(binary_corners)
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_corners)
            if ret == 5:
                centroids = np.int0(centroids[1:])
                break
            elif ret < 5:
                break
            else:
                dst = cv2.dilate(dst, None)

        try:
            inner, outer = self._identify_points(centroids)
        except (BaseNotFound, TypeError) as e:
            return None

        corners = [outer[0], inner[0], inner[1], outer[1]]
        corners = [bg.Point2D(point[0], point[1]) for point in corners]
        return quad.Quad(self._scale_to_quad_space(corners))
        # for line in lines:
        #     color = (random.uniform(50,255), random.uniform(50,255), random.uniform(50,255))
            # print("(" + str(x1) + "," + str(y1) + "),(" + str(x2) + "," + str(y2) + ")")
            # cv2.line(img, (line[0][0], line[0][1]), (line[1][0], line[1][1]), color, 2)
        # img[centroids[:, 1], centroids[:, 0]] = [0, 0, 255]

        # Threshold for an optimal value, it may vary depending on the image.
        # img[dst > 0.01 * dst.max()] = [0, 0, 255]

        # cv2.imshow('and', binary)
        # cv2.imshow('img', img)
        # cv2.imshow('dst', dst)
        # if cv2.waitKey(0) & 0xff == 27:
        #     cv2.destroyAllWindows()


def test():
    renderer = rend.Renderer(channels=3)
    detector = CornerQuadDetector()
    orig = quad.create_random(0.7, 0.2, 0.7, 0.2, 2)
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
