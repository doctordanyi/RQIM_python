import cv2
import numpy as np
import lib.utils.geometry as geom

lsd = cv2.createLineSegmentDetector()
img = cv2.imread("out/quad0.png", 0)

lines, widths, prec, nfa = lsd.detect(img)
# blank = np.zeros(img.shape, np.uint8)
# drw = lsd.drawSegments(blank, lines)
# cv2.imshow("detected", drw)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

mod_lines = np.zeros(lines.shape)
for line in lines:
    print(line[0])
    delta = line[0, 2:4] - line[0, 0:2]
    delta = (delta / np.linalg.norm(delta)) * 5/2
    delta = geom.rotate(delta, np.pi / 2)
    delta = np.concatenate((delta, delta))
    print(delta)
    mod_lines.append(line[0] + delta)

blank = np.zeros(img.shape, np.uint8)
drw = lsd.drawSegments(blank, mod_lines)
cv2.imshow("detected", drw)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(lines)
print(widths)
pass
