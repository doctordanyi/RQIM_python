"""Run RQIM experiments."""
import json, cv2
import lib.structures.quad as quad
import lib.graphics.renderer as renderer
import experiment.detector as detector
from bokeh.plotting import figure
from bokeh.io import show


rend = renderer.Renderer()
det = detector.LSDQuadDetector()
i = 0
error = []
while i < 100:
    q = quad.create_random(0.5, 0.2, 1, 0.3, 2)
    if q.is_valid():
        img = rend.render(q)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        q_det = det.detect_quad(img)
        if q_det is not None:
            err = q.abs_difference(q_det)
            error.append(err)

        i += 1

cv2.destroyAllWindows()
f = figure()
f.line(range(len(error)), error)
show(f)
