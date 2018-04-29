"""Run RQIM experiments."""
import cv2, numpy
import lib.structures.quad as quad
import lib.graphics.renderer as renderer
import lib.detectors.lsd as lsd_detector
import lib.detectors.hough as hough_detector
import lib.detectors.corner as corner_detector
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


gen = quad.QuadGenerator(quads_per_scale=1000, base_lengths=numpy.arange(0.01, 0.75, 0.01))
gen.generate()
gen.save_to_json("out/quads_generated.json")

# Define error mean and dev test steps
rel_coord = steps.GetErrorMeanAndDeviation(title="Corner: Relative Coordinate Error",
                                           out_file_name="corner_relative_coordinate_error",
                                           error_function=quad.get_rel_avg_position_error)

rel_angle = steps.GetErrorMeanAndDeviation(title="Corner: Relative Angle Error",
                                           out_file_name="corner_relative_angle_error",
                                           error_function=quad.get_rel_avg_angle_error)

rel_orient = steps.GetErrorMeanAndDeviation(title="Corner: Relative Orientation Error",
                                            out_file_name="corner_relative_orientation_error",
                                            error_function=quad.get_rel_orientation_error)

rel_multiplier = steps.GetErrorMeanAndDeviation(title="Corner: Relative Multiplier Error",
                                                out_file_name="corner_relative_multiplier_error",
                                                error_function=quad.get_rel_avg_multiplier_error)

rel_base_length = steps.GetErrorMeanAndDeviation(title="Corner: Relative Base length Error",
                                                 out_file_name="corner_relative_base_length_error",
                                                 error_function=quad.get_rel_base_length_error)

rec_unrec_count = steps.GetRecognitionCount(title="Corner: Recognition Statistics",
                                            out_file_name="corner_rec_unrec_count")

rend = renderer.Renderer(height=640, width=640, channels=3)
det = corner_detector.CornerQuadDetector()
exp = Experiment(gen, rend, det)
exp.steps.append(rec_unrec_count)
exp.steps.append(rel_coord)
exp.steps.append(rel_angle)
exp.steps.append(rel_orient)
exp.steps.append(rel_multiplier)
exp.steps.append(rel_base_length)
exp.run()

