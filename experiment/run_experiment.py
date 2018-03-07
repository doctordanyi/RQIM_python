"""Run RQIM experiments."""
import cv2, numpy
import lib.structures.quad as quad
import lib.graphics.renderer as renderer
import lib.detectors.lsd as detector
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import NumeralTickFormatter


class Experiment:
    def __init__(self):
        self.steps = []

    def add_step(self, step):
        self.steps.append(step)


gen = quad.QuadGenerator(quads_per_scale=10, base_lengths=numpy.arange(0.01, 0.6, 0.01))
gen.generate()
gen.save_to_json("out/quads_generated.json")

# rend = renderer.Renderer(height=640, width=640)
# det = detector.LSDQuadDetector()
# mean = []
# dev = []
# unrec = []
# scales = numpy.arange(0.6, 0.01, -0.01)
# for scale in scales:
#     error = []
#     i = 0
#     not_recognised = 0
#     print("Running test for scale " + str(scale))
#     while i < 100:
#         q = quad.create_random(scale, 0.2, 1, 0.3, 2)
#         if q.is_valid():
#             img = rend.render(q)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             q_det = det.detect_quad(img)
#             if q_det is not None:
#                 err = quad.get_error_rel_avg(q, q_det)
#                 if err > 1:
#                     not_recognised += 1
#                 else:
#                     error.append(err)
#             else:
#                 not_recognised += 1
#
#             i += 1
#     mean.append(numpy.mean(error))
#     dev.append(numpy.std(error))
#     unrec.append(not_recognised)
#
# mean = numpy.array(mean)
# dev = numpy.array(dev)
# f = figure(title="LSD Detector results", plot_width=1200, plot_height=600)
#
# f.xaxis.axis_label = "Base length"
# f.yaxis[0].formatter = NumeralTickFormatter(format="0.00%")
#
# f.line(scales, mean, legend="Error mean", line_color="red")
# f.circle(scales, mean, legend="Error mean", line_color="red", fill_color="red")
#
# f.square(scales, dev, legend="Deviation", fill_color=None, line_color="green")
# f.line(scales, dev, legend="Deviation", line_color="green")
#
# f.legend.location = "top_right"
#
# show(f)
