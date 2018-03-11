"""Run RQIM experiments."""
import cv2, numpy
import lib.structures.quad as quad
import lib.graphics.renderer as renderer
import lib.detectors.lsd as detector
import experiment.steps as steps


class Experiment:
    def __init__(self, quad_provider, renderer, detector):
        self.steps = []
        self.quad_provider = quad_provider
        self.renderer = renderer
        self.detector = detector

    def run(self):
        for group in self.quad_provider.get_quad_groups():
            # Do the detection for the current group
            print("Running test for quads grouped by " + group["group_by"] + ", with parameter: " + str(group["param"]))
            det_quads = [self.detector.detect_quad(self.renderer.render(q)) for q in group["quads"]]
            for step in self.steps:
                step.process_group(group["param"], group["quads"], det_quads)

        for step in self.steps:
            step.save_results()


gen = quad.QuadGenerator(quads_per_scale=250, base_lengths=numpy.arange(0.01, 0.75, 0.01))
gen.generate()
gen.save_to_json("out/quads_generated.json")

rend = renderer.Renderer(height=640, width=640, channels=1)
det = detector.LSDQuadDetector()
exp = Experiment(gen, rend, det)
exp.steps.append(steps.GetRelativeCoordinateError())
exp.steps.append(steps.GetRecognitionCount())
exp.steps.append(steps.GetRelativeAngleError())
exp.steps.append(steps.GetRelativeOrientationError())
exp.steps.append(steps.GetRelativeMultiplierError())
exp.run()

