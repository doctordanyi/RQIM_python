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
        super().__init__()

    def _init_iteration(self, img):
        self._orig_img = img
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._working_img = cv2.threshold(img, thresh=127, maxval=255, type=cv2.THRESH_BINARY_INV)[1]

        self._set_bounding_box_size()
        pass


    def _identify_points(self, centroids):
        binary = cv2.threshold(self._working_img, thresh=127, maxval=255, type=cv2.THRESH_BINARY_INV)[1]

        pointpairs = itertools.combinations(centroids.tolist(), 2)
        lines = []
        for pointpair in pointpairs:
            line_points = geom.createLineIterator(np.array(pointpair[0], dtype=np.int_), np.array(pointpair[1], dtype=np.int_), binary)
            hit_miss_ratio = np.count_nonzero(line_points[:,2]) / len(line_points)
            lines.append((hit_miss_ratio, pointpair))

        lines.sort(key=lambda line: line[0])
        lines = lines[0:3]
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


    def _merge_corners(self, corners):
        if len(corners) == 4:
            return corners

        corner_list = [corner[0] for corner in corners.tolist()]
        pairs = itertools.combinations(corner_list, 2)
        pairs = [(pair, geom.distance(pair[0], pair[1])) for pair in pairs]
        pairs.sort(key=lambda el: el[1])
        closest = pairs[0][0]

        corner_list.remove(closest[0])
        corner_list.remove(closest[1])
        corner_list.append([(closest[0][0] + closest[1][0]) / 2, (closest[0][1] + closest[1][1]) / 2])

        corner_list = [[corner] for corner in corner_list]
        return self._merge_corners(np.array(corner_list))

    def detect_quad(self, img):
        self._init_iteration(img)

        gray = np.float32(self._working_img)
        corners_good = cv2.goodFeaturesToTrack(gray, 6, 0.1, 2)
        # for i in corners_good:
        #     x, y = i.ravel()
        #     cv2.circle(self._orig_img, (int(x), int(y)), 2, (0,255,0), -1)

        if len(corners_good) < 4:
            return None
        corners_good = self._merge_corners(corners_good)
        # for i in corners_good:
        #     x, y = i.ravel()
        #     cv2.circle(self._orig_img, (int(x), int(y)), 2, (0,0,255), -1)

        corners_good = np.stack([corner[0] for corner in corners_good])

        try:
            inner, outer = self._identify_points(corners_good)
        except (BaseNotFound, TypeError) as e:
            print(e)
            return None

        corners = [outer[0], inner[0], inner[1], outer[1]]
        corners = [bg.Point2D(point[0], point[1]) for point in corners]
        # for i in corners:
        #     x, y = i
        #     cv2.circle(self._orig_img, (int(x), int(y)), 2, (255,0,0), -1)
        #
        # cv2.imshow('img', self._orig_img)
        # cv2.waitKey(0)
        return quad.Quad(self._scale_to_quad_space(corners))


def test():
    renderer = rend.Renderer(channels=3)
    detector = CornerQuadDetector()
    orig = quad.create_random(0.4, 0.2, 0.7, 0.2, 2)
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
