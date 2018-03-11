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


class GetErrorMeanAndDeviation(TestStep):
    def __init__(self, title, out_file_name, error_function):
        self.scales = []
        self.mean = []
        self.deviation = []
        self._figure = None
        self._error_function = error_function
        self._out_file_name = out_file_name
        self._title = title

    def process_group(self, param, orig_quads, det_quads):
        self.scales.append(param)
        if det_quads.count(None) > (len(det_quads) - 5):
            self.mean.append(np.NaN)
            self.deviation.append(np.NaN)
        else:
            error = [self._error_function(q, det) for (q, det) in zip(orig_quads, det_quads) if det is not None]
            # if the error is greater than 100%, do not count it as detected, remove from the list
            error = [err for err in error if err < 1]
            self.mean.append(np.mean(error))
            self.deviation.append(np.std(error))
            pass

    def _prepare_plot(self):
        mean = np.array(self.mean)
        deviation = np.array(self.deviation)

        self._figure = bp.figure(title=self._title, plot_width=1200, plot_height=600)

        self._figure.xaxis.axis_label = "Base length"
        self._figure.yaxis[0].formatter = bm.NumeralTickFormatter(format="0.00%")

        self._figure.line(self.scales, mean, legend="Error mean", line_color="red")
        self._figure.circle(self.scales, mean, legend="Error mean", line_color="red", fill_color="red")

        self._figure.square(self.scales, deviation, legend="Deviation", fill_color=None, line_color="green")
        self._figure.line(self.scales, deviation, legend="Deviation", line_color="green")

        self._figure.legend.location = "top_right"

    def save_results(self):
        if self._figure is None:
            self._prepare_plot()

        bi.output_file("out/plots/" + self._out_file_name + ".html", title=self._title)
        bi.save(self._figure)

    def show_results(self):
        if self._figure is None:
            self._prepare_plot()

        bi.show(self._figure)


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