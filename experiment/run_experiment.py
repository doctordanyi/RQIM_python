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

# Define error mean and dev test steps
rel_coord = steps.GetErrorMeanAndDeviation(title="LSD: Relative Coordinate Error",
                                           out_file_name="lsd_relative_coordinate_error",
                                           error_function=quad.get_rel_avg_position_error)

rel_angle = steps.GetErrorMeanAndDeviation(title="LSD: Relative Angle Error",
                                           out_file_name="lsd_relative_angle_error",
                                           error_function=quad.get_rel_avg_angle_error)

rel_orient = steps.GetErrorMeanAndDeviation(title="LSD: Relative Orientation Error",
                                            out_file_name="lsd_relative_orientation_error",
                                            error_function=quad.get_rel_orientation_error)

rel_multiplier = steps.GetErrorMeanAndDeviation(title="LSD: Relative Multiplier Error",
                                                out_file_name="lsd_relative_multiplier_error",
                                                error_function=quad.get_rel_avg_multiplier_error)

rend = renderer.Renderer(height=640, width=640, channels=1)
det = detector.LSDQuadDetector()
exp = Experiment(gen, rend, det)
exp.steps.append(steps.GetRecognitionCount())
exp.steps.append(rel_coord)
exp.steps.append(rel_angle)
exp.steps.append(rel_orient)
exp.steps.append(rel_multiplier)
exp.run()

