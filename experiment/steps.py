import numpy as np
import bokeh.plotting as bp
import bokeh.io as bi
import bokeh.models as bm
import lib.structures.quad as quad


class TestStep:
    def process_group(self, param, orig_quads, det_quads):
        raise NotImplemented

    def save_results(self):
        raise NotImplemented

    def show_results(self):
        raise NotImplemented


class GetRelativeCoordinateError(TestStep):
    def __init__(self):
        self.scales = []
        self.mean = []
        self.deviation = []
        self.figure = None

    def process_group(self, param, orig_quads, det_quads):
        self.scales.append(param)
        if det_quads.count(None) > (len(det_quads) - 5):
            self.mean.append(np.NaN)
            self.deviation.append(np.NaN)
        else:
            error = [quad.get_rel_avg_position_error(q, det) for (q, det) in zip(orig_quads, det_quads) if det is not None]
            # if the error is greater than 100%, do not count it as detected, remove from the list
            error = [err for err in error if err < 1]
            self.mean.append(np.mean(error))
            self.deviation.append(np.std(error))
            pass

    def _prepare_plot(self):
        mean = np.array(self.mean)
        deviation = np.array(self.deviation)

        self.figure = bp.figure(title="LSD Detector results", plot_width=1200, plot_height=600)

        self.figure.xaxis.axis_label = "Base length"
        self.figure.yaxis[0].formatter = bm.NumeralTickFormatter(format="0.00%")

        self.figure.line(self.scales, mean, legend="Error mean", line_color="red")
        self.figure.circle(self.scales, mean, legend="Error mean", line_color="red", fill_color="red")

        self.figure.square(self.scales, deviation, legend="Deviation", fill_color=None, line_color="green")
        self.figure.line(self.scales, deviation, legend="Deviation", line_color="green")

        self.figure.legend.location = "top_right"

    def save_results(self):
        if self.figure is None:
            self._prepare_plot()

        bi.output_file("out/plots/relative_coordinate_error.html", title="Relative coordinate error")
        bi.save(self.figure)

    def show_results(self):
        if self.figure is None:
            self._prepare_plot()

        bi.show(self.figure)


class GetRecognitionCount(TestStep):
    def __init__(self):
        self.recognised = []
        self.unrecognised = []
        self.scales = []
        self.figure = None

    def process_group(self, param, orig_quads, det_quads):
        self.scales.append(param)
        self.recognised.append(len(det_quads) - det_quads.count(None))
        self.unrecognised.append(det_quads.count(None))

    def _prepare_plot(self):
        recognised = np.array(self.recognised)
        unrecognised = np.array(self.unrecognised)

        self.figure = bp.figure(title="LSD Detector results", plot_width=1200, plot_height=600)

        self.figure.xaxis.axis_label = "Base length"
        self.figure.yaxis.axis_label = "Count"

        self.figure.line(self.scales, recognised, legend="Recognised", line_color="blue")
        self.figure.circle(self.scales, recognised, legend="Recognised", line_color="blue", fill_color="blue")

        self.figure.square(self.scales, unrecognised, legend="Unrecognised", fill_color=None, line_color="orange")
        self.figure.line(self.scales, unrecognised, legend="Unrecognised", line_color="orange")

        self.figure.legend.location = "top_right"

    def save_results(self):
        if self.figure is None:
            self._prepare_plot()

        bi.output_file("out/plots/recognised_unrecognised_count.html", title="Recognised count")
        bi.save(self.figure)

    def show_results(self):
        if self.figure is None:
            self._prepare_plot()

        bi.show(self.figure)


class GetRelativeAngleError(TestStep):
    def __init__(self):
        self.scales = []
        self.mean = []
        self.deviation = []
        self.figure = None

    def process_group(self, param, orig_quads, det_quads):
        self.scales.append(param)
        if det_quads.count(None) > (len(det_quads) - 5):
            self.mean.append(np.NaN)
            self.deviation.append(np.NaN)
        else:
            error = [quad.get_rel_avg_angle_error(q, det) for (q, det) in zip(orig_quads, det_quads) if det is not None]
            # if the error is greater than 100%, do not count it as detected, remove from the list
            error = [err for err in error if err < 1]
            self.mean.append(np.mean(error))
            self.deviation.append(np.std(error))
            pass

    def _prepare_plot(self):
        mean = np.array(self.mean)
        deviation = np.array(self.deviation)

        self.figure = bp.figure(title="LSD Detector: relative average angle error", plot_width=1200, plot_height=600)

        self.figure.xaxis.axis_label = "Base length"
        self.figure.yaxis[0].formatter = bm.NumeralTickFormatter(format="0.00%")

        self.figure.line(self.scales, mean, legend="Error mean", line_color="red")
        self.figure.circle(self.scales, mean, legend="Error mean", line_color="red", fill_color="red")

        self.figure.square(self.scales, deviation, legend="Deviation", fill_color=None, line_color="green")
        self.figure.line(self.scales, deviation, legend="Deviation", line_color="green")

        self.figure.legend.location = "top_right"

    def save_results(self):
        if self.figure is None:
            self._prepare_plot()

        bi.output_file("out/plots/relative_angle_error.html", title="Relative angle error")
        bi.save(self.figure)

    def show_results(self):
        if self.figure is None:
            self._prepare_plot()

        bi.show(self.figure)


class GetRelativeOrientationError(TestStep):
    def __init__(self):
        self.scales = []
        self.mean = []
        self.deviation = []
        self.figure = None

    def process_group(self, param, orig_quads, det_quads):
        self.scales.append(param)
        if det_quads.count(None) > (len(det_quads) - 5):
            self.mean.append(np.NaN)
            self.deviation.append(np.NaN)
        else:
            error = [quad.get_rel_orientation_error(q, det) for (q, det) in zip(orig_quads, det_quads) if det is not None]
            # if the error is greater than 100%, do not count it as detected, remove from the list
            error = [err for err in error if err < 1]
            self.mean.append(np.mean(error))
            self.deviation.append(np.std(error))
            pass

    def _prepare_plot(self):
        mean = np.array(self.mean)
        deviation = np.array(self.deviation)

        self.figure = bp.figure(title="LSD Detector: relative orientation error", plot_width=1200, plot_height=600)

        self.figure.xaxis.axis_label = "Base length"
        self.figure.yaxis[0].formatter = bm.NumeralTickFormatter(format="0.00%")

        self.figure.line(self.scales, mean, legend="Error mean", line_color="red")
        self.figure.circle(self.scales, mean, legend="Error mean", line_color="red", fill_color="red")

        self.figure.square(self.scales, deviation, legend="Deviation", fill_color=None, line_color="green")
        self.figure.line(self.scales, deviation, legend="Deviation", line_color="green")

        self.figure.legend.location = "top_right"

    def save_results(self):
        if self.figure is None:
            self._prepare_plot()

        bi.output_file("out/plots/relative_orientation_error.html", title="Relative orientation error")
        bi.save(self.figure)

    def show_results(self):
        if self.figure is None:
            self._prepare_plot()

        bi.show(self.figure)


class GetRelativeMultiplierError(TestStep):
    def __init__(self):
        self.scales = []
        self.mean = []
        self.deviation = []
        self.figure = None

    def process_group(self, param, orig_quads, det_quads):
        self.scales.append(param)
        if det_quads.count(None) > (len(det_quads) - 5):
            self.mean.append(np.NaN)
            self.deviation.append(np.NaN)
        else:
            error = [quad.get_rel_avg_multiplier_error(q, det) for (q, det) in zip(orig_quads, det_quads) if det is not None]
            # if the error is greater than 100%, do not count it as detected, remove from the list
            error = [err for err in error if err < 1]
            self.mean.append(np.mean(error))
            self.deviation.append(np.std(error))
            pass

    def _prepare_plot(self):
        mean = np.array(self.mean)
        deviation = np.array(self.deviation)

        self.figure = bp.figure(title="LSD Detector: relative multiplier error", plot_width=1200, plot_height=600)

        self.figure.xaxis.axis_label = "Base length"
        self.figure.yaxis[0].formatter = bm.NumeralTickFormatter(format="0.00%")

        self.figure.line(self.scales, mean, legend="Error mean", line_color="red")
        self.figure.circle(self.scales, mean, legend="Error mean", line_color="red", fill_color="red")

        self.figure.square(self.scales, deviation, legend="Deviation", fill_color=None, line_color="green")
        self.figure.line(self.scales, deviation, legend="Deviation", line_color="green")

        self.figure.legend.location = "top_right"

    def save_results(self):
        if self.figure is None:
            self._prepare_plot()

        bi.output_file("out/plots/relative_multiplier_error.html", title="Relative multiplier error")
        bi.save(self.figure)

    def show_results(self):
        if self.figure is None:
            self._prepare_plot()

        bi.show(self.figure)

