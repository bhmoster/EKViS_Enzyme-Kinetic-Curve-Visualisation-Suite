import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tools.michaelis_menten import MichaelisMenten
from tools.utility import Unit, UtilityUnit

cur_dir = Path(".")


class Config:
    """This class handles parsing of cmd input."""

    def __init__(self):
        self._args = sys.argv
        self._data_dir = cur_dir / "data"
        self._res_dir = cur_dir / "results"
        self._unit = UtilityUnit.Micro
        self._extinction_coefficient = 0.021
        self._time_coefficient = 95 / 60
        self._save_fig = False
        self._log_scale = False
        self._plotter = None
        self._title = "Title Not Specified"

        self._parseArguments(sys.argv)

        if not self._res_dir.exists():
            os.makedirs(self._res_dir)

    def GetDataDir(self):
        return self._data_dir

    def GetResultsDir(self):
        return self._res_dir

    def GetUnit(self):
        return self._unit

    def GetExtinctionCoefficient(self):
        return self._extinction_coefficient

    def GetTimeCoefficient(self):
        return self._time_coefficient

    def GetSaveMode(self):
        return self._save_fig

    def GetLogMode(self):
        return self._log_scale

    def GetPlotter(self):
        return self._plotter

    def GetPlotTitle(self):
        return self._title

    @staticmethod
    def GetHelpText():
        return """
Usage:
    curve_plotter.py [options] <action>

    action:
        MichaelisMenten

    options:
        -u | --unit  <unit specifier>   => the unit used, default=micro
        -t | --title <plot title>       => the title of plot being created
        -l | --logarithmic              => set x-axis to logarithmic scale
        -s | --save                     => whether to save the plot, default=show

    unit specifier:
        femto, pico, nano, micro, milli, molar

    example use:
        curve_plotter.py --save --unit "milli" --title "EnzymeXYZ plotted for DATA" --logarithmic MichaelisMenten

        This will save a created Michaelis-Menton plot, with the specified title, using milli moles as the unit and with a logarithmic scale set to the x-axis.
"""

    def _parseArguments(self, args):
        mode = args[-1]
        options = args[1:-1]
        i = 0
        lim = len(options)

        while i < lim:
            match options[i]:
                case "--save" | "-s":
                    self._save_fig = True
                    i += 1

                case "--unit" | "-u":
                    self._unit = UtilityUnit.from_text(options[i + 1])
                    i += 2

                case "--title" | "-t":
                    self._title = options[i + 1]
                    i += 2

                case "--logarithmic" | "-l":
                    self._log_scale = True
                    i += 1

                case _:
                    raise Exception("Could not decide on option.")

        match mode:
            case "MichaelisMenten":
                self._plotter = MichaelisMenten(self._unit)

            case _:
                raise Exception("Could not decide on plotter.")


class DataReader:
    def __init__(self, config):
        self._config: Config = config
        

    def _handle_column(self, col: [float]) -> Unit:
        """
        Handles a single replicate column:
        - normalizes absorbance
        - determines the maximally linear initial window
        - computes the initial rate from that window
        """

        # --- parameters you may want to tweak ---
        min_points = 4           # minimum points to attempt a fit
        r2_threshold = 0.98      # linearity criterion 

        # --- normalize absorbance ---
        col = np.asarray(col, dtype=float)
        col_norm = col - col[0]

        # time axis in seconds
        dt = 5.0  # seconds
        t = np.arange(len(col_norm)) * dt

        # --- adaptive linear window ---
        best_n = min_points
        best_slope = None

        for n in range(min_points, len(col_norm) + 1):
            x = t[:n]
            y = col_norm[:n]

            # linear fit: y = m x + b
            m, b = np.polyfit(x, y, 1)
            y_fit = m * x + b

            # compute R²
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

                # guard against flat traces
            if ss_tot == 0:
                break

            r2 = 1 - ss_res / ss_tot

            if r2 < r2_threshold:
                break

            best_n = n
            best_slope = m

            # fallback (should almost never trigger)
        if best_slope is None:
            best_slope, _ = np.polyfit(t[:min_points], col_norm[:min_points], 1)

        # --- convert slope to initial velocity ---
        # slope is Abs/sec → convert to concentration/min
        init_velocity = (
            best_slope * 60 / config.GetExtinctionCoefficient()
        )

        return init_velocity

    def handle_file(self, file_path: str, config: Config) -> (Unit, [float]):
        """ handle_file Handles a data file. """

        # extract concentration from file name
        concentration = float(
            re.search(r"(\d+(_(\d+))?)", str(file_path)).group(0).replace("_", ".")
        )

        # read data and compute initial velocity
        df = pd.read_csv(file_path, header=None)

        reduction_cytochromeC = df.apply(self._handle_column, axis=0)
        
        return Unit(concentration), reduction_cytochromeC



if __name__ == "__main__":
    if sys.argv[1] in ["-h", "--help"]:
        print(Config.GetHelpText())
        exit()

    config = Config()
    reader = DataReader(config)

    data = {}

    for data_file in config.GetDataDir().glob("*"):
        concentration, velocity = reader.handle_file(data_file, config)

        data[concentration.get_num(config.GetUnit())] = velocity

    data = dict(sorted(data.items()))

    plotter = config.GetPlotter()
    plotter.make_plot(
        data,
        config.GetResultsDir(),
        config.GetPlotTitle(),
        config.GetSaveMode(),
        config.GetLogMode(),
    )
