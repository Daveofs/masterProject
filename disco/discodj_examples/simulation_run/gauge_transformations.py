import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as onp

from discodj.core.array_file_utils import tree_scalar_flatten
from discodj.cosmology.cosmology import Cosmology
from discodj.gauge_transformations import LinearGRTransferTable, compute_linear_gr_transfer_from_discoeb
from .config_state import get_global_state_from_args, ConfigState
from .cosmology import cosmology_from_config
from .run_utils import compile_run_and_profile


def make_64bit(input):
    if isinstance(input, jax.Array):
        if input.dtype == jnp.float32:
            return jnp.astype(input, jnp.float64)
        if input.dtype == jnp.int32:
            return jnp.astype(input, jnp.int64)
    return input


def create_gr_transfer_in_file(state: ConfigState, cosmo: Cosmology, file: Path):
    print(f"generating from a={state.a_ini} to a={state.a_end}")
    a_table = jnp.geomspace(state.a_ini, state.a_end, 64)

    import matplotlib.pyplot as plt
    import asdf
    with jax.enable_x64():
        cosmo = jax.tree.map(make_64bit, cosmo)
        if cosmo.Omega_r:
            raise NotImplemented("no radiation")

        grtable = compile_run_and_profile(
            compute_linear_gr_transfer_from_discoeb,
            "compute_linear_gr_transfer_from_discoeb",
            cosmo=cosmo,
            a=a_table
        )
    grtable = jax.tree.map(lambda arr: onp.asarray(arr), grtable)
    data = {
        "grtable": dataclasses.asdict(grtable),
        "cosmology": tree_scalar_flatten(dataclasses.asdict(cosmo))
    }
    af = asdf.AsdfFile(data)
    af.write_to(file)
    print(f"written to {file}")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ai in [0, len(grtable.a) // 2, -1]:
        axs[0].loglog(grtable.k, onp.abs(grtable.HT[:, ai]), label=f"a={grtable.a[ai]:.3f}")
        axs[1].loglog(grtable.k, onp.abs(grtable.HTprime[:, ai]), label=f"a={grtable.a[ai]:.3f}")

    axs[0].set_title("|H_T(k,a)|")
    axs[1].set_title("|H_T'(k,a)|")
    for ax in axs:
        ax.set_xlabel(r"$k$ [$h$/Mpc]")
        ax.set_ylabel("amplitude")
        ax.legend()
        ax.grid(True, which="both", ls=":", alpha=0.4)
    plt.tight_layout()
    plt.show()


def load_gr_transfer_from_file(file: Path) -> tuple[LinearGRTransferTable, Cosmology]:
    import asdf

    with asdf.open(file) as fd:
        grtable_data = fd["grtable"]
        grtable = LinearGRTransferTable(**grtable_data)
        grtable = jax.tree.map(lambda arr: jax.numpy.asarray(arr), grtable)

        cosmology_data = fd["cosmology"]
        cosmo = Cosmology(**cosmology_data)
    return grtable, cosmo


def grtable_main():
    state = get_global_state_from_args()
    cosmo = cosmology_from_config(state)

    if state.gr_correction_file is None:
        raise ValueError("gr_correction_file needs to be specified as output file")

    create_gr_transfer_in_file(state, cosmo, Path(state.gr_correction_file))
