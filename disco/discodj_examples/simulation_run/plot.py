from pathlib import Path

import numpy as onp
from matplotlib import pyplot as plt

from .config_state import ConfigState
from .run_utils import global_timing_stats
from .extra_dataclasses import PowerSpectrum


def plot_slices(state: ConfigState, output_dir: Path, delta_slices, delta_lc_slices):
    delta_slices_main = delta_lc_slices if state.lightcone else delta_slices
    global_timing_stats.start_timing("plotting slices")
    fig, axs = plt.subplots(1, 3, figsize=(10, 5), layout="constrained")
    axs[0].imshow(onp.log10(1.01 + delta_slices_main.mean))
    axs[0].set_title("Avg.")
    axs[1].imshow(onp.log10(1.01 + delta_slices_main.slice))
    axs[1].set_title("Slice")
    axs[2].imshow(onp.log10(1.01 + delta_slices_main.thin_slice))
    axs[2].set_title("Thin Slice")
    fig.savefig(output_dir / "comparison.pdf")

    plt.imsave(output_dir / "mean.png", onp.log10(1.01 + delta_slices.mean))
    plt.imsave(output_dir / "slice.png", onp.log10(1.01 + delta_slices.slice))
    plt.imsave(output_dir / "thinslice.png", onp.log10(1.01 + delta_slices.thin_slice))
    if state.lightcone:
        plt.imsave(output_dir / "mean_lc.png", onp.log10(1.01 + delta_lc_slices.mean))
        plt.imsave(output_dir / "slice_lc.png", onp.log10(1.01 + delta_lc_slices.slice))
        plt.imsave(output_dir / "thinslice_lc.png", onp.log10(1.01 + delta_lc_slices.thin_slice))

    global_timing_stats.stop_timing("plotting slices")


def plot_powerspectrum(output_dir: Path, ps: PowerSpectrum):
    global_timing_stats.start_timing("plotting powerspectrum")

    fig = plt.figure()
    ax = fig.gca()

    ax.loglog(ps.k, ps.P)
    ax.set_xlabel(r"$k$ [$h$/Mpc]")
    ax.set_ylabel(r"$P(k)$ [$h^{-3}$Mpc$^3$]")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    # TODO: Nicer plot

    fig.savefig(output_dir / "powerspectrum.pdf")

    global_timing_stats.stop_timing("plotting powerspectrum")
