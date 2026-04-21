import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tools.utility import UtilityUnit, compute_standard_error


class MichaelisMenten:
    def __init__(self, unit: UtilityUnit):
        self._unit = unit

    def _michaelis_menten(self, S, Vmax, Km):
        """Calculates the Michaelis-Menten function."""

        return (Vmax * S) / (Km + S)

    def _extract_parameters(self, concentrations, velocities):
        """Computes Km and Vmax per experiment repetition."""

        vmax = []
        km = []
        kcat = []
        kcat_km = []

        # compute parameters per column
        for i in range(velocities.shape[1]):
            params, _ = curve_fit(
                self._michaelis_menten,
                concentrations,
                velocities[:, i],
                p0=[max(velocities[:, i]), np.median(concentrations)],
            )

            # compute the relevant parameters
            Vmax, Km = params
            Kcat = Vmax / 0.01

            vmax.append(Vmax)
            km.append(Km)
            kcat.append(Kcat)
            kcat_km.append(Kcat / Km)

        print(f"Vmax: {np.mean(vmax):.4f} ± {compute_standard_error(vmax):.4f}")
        print(f"Km: {np.mean(km):.4f} ± {compute_standard_error(km):.4f}")
        print(f"Kcat: {np.mean(kcat):.4f} ± {compute_standard_error(kcat):.4f}")
        print(
            f"Kcat/Km: {np.mean(kcat_km):.4f} ± {compute_standard_error(kcat_km):.4f}"
        )

    def make_plot(self, data, res_dir, title, save, logarithmic):
        # split data into concentrations and velocities
        substrate_concentrations = np.array(list(data.keys()))
        initial_velocities = np.array(list(data.values()))

        # compute and print Km and Vmax average+deviation
        self._extract_parameters(substrate_concentrations, initial_velocities)

        # prepare data for plot
        initial_velocities = np.array([entry.mean() for entry in data.values()])
        errors = np.array([entry.std() for entry in data.values()])

        # fit model to data
        (Vmax, Km), _ = curve_fit(
            self._michaelis_menten,
            substrate_concentrations,
            initial_velocities,
            p0=[max(initial_velocities), np.median(substrate_concentrations)],
        )

        # plot the michaelis-menten curve
        S_fit = np.linspace(
            min(substrate_concentrations), max(substrate_concentrations), 100
        )
        v_fit = self._michaelis_menten(S_fit, Vmax, Km)

        plt.figure(figsize=(8, 6))
        plt.errorbar(
            substrate_concentrations,
            initial_velocities,
            yerr=errors,
            fmt="o",
            ecolor="black",
            markerfacecolor="black",
            markeredgecolor="black",
            label="Experimental data ± SD",
        )
        plt.plot(S_fit, v_fit, label="Michaelis-Menten fit", color="black")
        plt.xlabel(f"[S] ({UtilityUnit.get_text(self._unit)})")
        plt.ylabel(
            f"Cytochrome c reductase activity ({UtilityUnit.get_text(self._unit)} / min)"
        )
        plt.title(title)
        # plt.legend()
        plt.grid(True)

        if logarithmic:
            plt.xscale("log")

        if save:
            plt.savefig(res_dir / "MichaelisMenten.png")
        else:
            plt.show()
