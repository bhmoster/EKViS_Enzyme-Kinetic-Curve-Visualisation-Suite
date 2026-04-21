import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t
from tools.utility import UtilityUnit, compute_standard_error, format_with_uncertainty
import matplotlib as mpl

mpl.rcParams["ps.fonttype"] = 42   # TrueType

class MichaelisMenten:
    def __init__(self, unit: UtilityUnit):
        self._unit = unit

    def _michaelis_menten(self, S, Vmax, Km):
        """Calculates the Michaelis-Menten function."""

        return (Vmax * S) / (Km + S)

    def _extract_parameters(self, concentrations, velocities, weighted=True):
        """
        Computes Michaelis–Menten parameters.
        If weighted=True, uses weighted nonlinear regression 
        Returns: Vmax, Km, and optionally their errors for reporting.
        """
        means = velocities.mean(axis=1)
        stds = velocities.std(axis=1, ddof=1)
        stds[stds == 0] = np.min(stds[stds > 0])  # guard against zero variance

        if weighted:
            sigma = stds
            absolute_sigma = True
        else:
            sigma = None
            absolute_sigma = False

        params, cov = curve_fit(
            self._michaelis_menten,
            concentrations,
            means,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            p0=[np.max(means), np.median(concentrations)],
        )

        Vmax, Km = params
        if weighted:
            Vmax_err, Km_err = np.sqrt(np.diag(cov))
            # 95% CI
            dof = len(concentrations) - len(params)
            tval = t.ppf(0.975, dof)
            Vmax_ci = tval * Vmax_err
            Km_ci = tval * Km_err

            # derived parameters
            enzyme_conc = 0.01  # µM
            kcat = Vmax / enzyme_conc
            kcat_err = Vmax_err / enzyme_conc
            kcat_ci = tval * kcat_err

            kcat_km = kcat / Km
            kcat_km_err = kcat_km * np.sqrt(
                (kcat_err / kcat) ** 2 + (Km_err / Km) ** 2
            )
            kcat_km_ci = tval * kcat_km_err

            # reporting
            print("Weighted Michaelis–Menten parameters (95% CI):")
            print(f"Vmax     = {format_with_uncertainty(Vmax, Vmax_err)}  (±{format_with_uncertainty(Vmax_ci, Vmax_ci)})")
            print(f"Km       = {format_with_uncertainty(Km, Km_err)}      (±{format_with_uncertainty(Km_ci, Km_ci)})")
            print(f"kcat     = {format_with_uncertainty(kcat, kcat_err)} (±{format_with_uncertainty(kcat_ci, kcat_ci)})")
            print(f"kcat/Km  = {format_with_uncertainty(kcat_km, kcat_km_err)} (±{format_with_uncertainty(kcat_km_ci, kcat_km_ci)})")

        return Vmax, Km

    def make_plot(self, data, res_dir, title, save, logarithmic):
        substrate_concentrations = np.array(list(data.keys()), dtype=float)
        initial_velocities = np.array(list(data.values()), dtype=float)

        # per-concentration mean and SD
        means = np.array([entry.mean() for entry in data.values()], dtype=float)
        stds = np.array([entry.std(ddof=1) for entry in data.values()], dtype=float)

        # flatten raw replicate points
        raw_S, raw_V = [], []
        for S, velocities in data.items():
            for v in np.asarray(velocities).ravel():
                raw_S.append(float(S))
                raw_V.append(float(v))
        raw_S = np.array(raw_S, dtype=float)
        raw_V = np.array(raw_V, dtype=float)

        # --- Weighted fit for reporting ---
        Vmax_w, Km_w = self._extract_parameters(substrate_concentrations, initial_velocities, weighted=True)

        # --- Unweighted fit for plotting ---
        Vmax_plot, Km_plot = self._extract_parameters(substrate_concentrations, initial_velocities, weighted=False)

        # prepare fit curve
        S_min = max(min(substrate_concentrations), 1e-12)
        S_max = max(substrate_concentrations)
        S_fit = np.logspace(np.log10(S_min), np.log10(S_max), 300) if logarithmic else np.linspace(S_min, S_max, 300)
        v_fit = self._michaelis_menten(S_fit, Vmax_plot, Km_plot)

        # plotting
        plt.figure(figsize=(8, 6))
        plt.scatter(raw_S, raw_V, s=30, alpha=0.8, label="Replicates", zorder=1)
        plt.errorbar(substrate_concentrations, means, yerr=stds,
                     fmt="o", color="black", ecolor="black",
                     markerfacecolor="white", markeredgecolor="black",
                     markersize=8, capsize=3, label="Mean ± SD", zorder=2)
        plt.plot(S_fit, v_fit, label="Michaelis–Menten fit", color="black", zorder=3)

        plt.xlabel(f"[S] ({UtilityUnit.get_text(self._unit)})")
        plt.ylabel(f"Cytochrome c reductase activity ({UtilityUnit.get_text(self._unit)} / min)")
        plt.title(title)
        plt.grid(True)
        if logarithmic:
            plt.xscale("log")
            plt.legend(loc = "upper left")
        else:
            plt.legend(loc = "lower right")

        if save:
            output_path = res_dir / f"{title}.eps"
            plt.savefig(output_path, format="eps", bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        # return both weighted and unweighted parameters for comparison
        return {
            "weighted": {"Vmax": Vmax_w, "Km": Km_w},
            "unweighted": {"Vmax": Vmax_plot, "Km": Km_plot}
        }