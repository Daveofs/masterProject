import dataclasses
from pathlib import Path

import jax
from jax import numpy as jnp
from jax_array_info import print_array_stats

from discodj import DiscoDJ
from discodj.core.array_file_utils import save_all_arrays_to_file
from discodj.core.scatter_and_gather import ScatterGatherProperties
from discodj.multidevice.communication import reorder_to_devices
from discodj.multidevice.utils import get_sharding_none
from discodj.nbody.nbody_dataclass import Particles, Positions

from .config_state import ConfigState
from .extra_dataclasses import FieldSlices, PowerSpectrum
from .lightcone_utils import lightcone_output_dataclass
from .plot import plot_powerspectrum, plot_slices
from .run_utils import compile_run_and_profile, global_timing_stats


def field_slices(field: jax.numpy.ndarray, state: ConfigState) -> FieldSlices:
    slice_thickness = state.res_pm // 128
    if slice_thickness < 2:
        slice_thickness = 2
    half_slice = slice_thickness // 2
    slice_layer = state.res_pm // 2
    if state.slice_axis == 0:
        field_slice = field[slice_layer, :, :]
        thin_slice = field[slice_layer - half_slice:slice_layer + half_slice, :, :].mean(0)
    elif state.slice_axis == 1:
        field_slice = field[:, slice_layer, :]
        thin_slice = field[:, slice_layer - half_slice:slice_layer + half_slice, :].mean(1)
    elif state.slice_axis == 2:
        field_slice = field[:, :, slice_layer]
        thin_slice = field[:, :, slice_layer - half_slice:slice_layer + half_slice].mean(2)
    else:
        raise ValueError("Invalid slice axis")
    slices = FieldSlices(
        mean=field.mean(state.slice_axis),
        slice=field_slice,
        thin_slice=thin_slice,
        slice_thickness=slice_thickness,
        slice_layer=slice_layer,
        slice_range=(slice_layer - half_slice,slice_layer + half_slice)
    )
    return jax.reshard(slices, get_sharding_none())

def delta_subset_lightcone(dj: DiscoDJ, sim_out: Particles, state: ConfigState,scatter_gather_props: ScatterGatherProperties):
    lightcone_particles, is_in_lightcone = lightcone_output_dataclass(state, dj, sim_out, compact_array=True)
    if state.padded_sim:
        lightcone_particles = reorder_to_devices(lightcone_particles, state.worder)

    delta_sim = dj.compute_field_quantity_from_particles(
        lightcone_particles, quantity=lightcone_particles.not_nan_mask(),
        normalize_by_density=False,
        antialias=0, deconvolve=False,
        res=state.res_pm, chunk_size=state.chunk_size,
        try_to_jit=False, scatter_gather_props=scatter_gather_props
    )
    slices = field_slices(delta_sim,state)
    if state.save_final_field:
        return slices, delta_sim, POS, is_in_lightcone
    return slices

def delta_subset(dj: DiscoDJ, sim_output: Particles, state: ConfigState, scatter_gather_props: ScatterGatherProperties):
    quantity = None
    if state.heft:
        quantity = sim_output.heft_data.f

    delta = dj.compute_field_quantity_from_particles(
        sim_output, res=state.res_pm, chunk_size=state.chunk_size, worder=state.worder,
        quantity=quantity, normalize_by_density=False,
        try_to_jit=False, scatter_gather_props=scatter_gather_props, deconvolve=state.deconvolve_density_field,
    )
    if quantity is None:
        delta = delta + 1  # TODO: implement properly like in _grad code
    slices = field_slices(delta,state)

    ps = None
    if state.powerspectrum:
        k, Pk_data, _ = dj.evaluate_power_spectrum(delta, deconvolve=False, worder=2, bins=32,
                                                    compute_std=False, try_to_jit=False)
        ps = PowerSpectrum(k, Pk_data)
    if not state.save_final_field:
        delta = None
    return slices, delta, ps


def density_calc(state:ConfigState, dj: DiscoDJ, sim_out: Particles, scatter_gather_props: ScatterGatherProperties, save_data, output_dir):
    global_timing_stats.start_timing("postprocessing")
    if state.lightcone:
        out_lc = compile_run_and_profile(
            delta_subset_lightcone,
            "delta_subset_lightcone",
            jit_arguments={"static_argnames": ("state","scatter_gather_props")},
            dj=dj,
            sim_out=sim_out,
            state=state,
            scatter_gather_props=scatter_gather_props

        )

        if state.save_final_field:
            delta_lc_slices, delta_full_lc, particle_pos_lc, is_in_lightcone = out_lc
        else:
            delta_lc_slices: FieldSlices = out_lc
        save_data["delta_lc_slices"] = delta_lc_slices.as_dict()
    else:
        delta_lc_slices = None
    delta_slices, delta_full, ps = compile_run_and_profile(
        delta_subset,
        "delta_subset",
        jit_arguments={"static_argnames": ("state","scatter_gather_props")},
        dj=dj,
        sim_output=sim_out,
        state=state,
        scatter_gather_props=scatter_gather_props
    )
    save_data["delta_slices"] = delta_slices.as_dict()
    global_timing_stats.stop_timing("postprocessing")

    plot_slices(state, output_dir, delta_slices, delta_lc_slices)
    if state.powerspectrum:
        save_data["power_spectrum"] = dataclasses.asdict(ps)
        plot_powerspectrum(output_dir, ps)


def save_slice_callback(idx, part: Particles, state: ConfigState, dj: DiscoDJ, scatter_gather_props: ScatterGatherProperties, output_dir:Path) -> None:
    assert not state.save_final_field
    snapshot_data = {}
    timestep_dir = output_dir / "steps"
    timestep_dir.mkdir(exist_ok=True)


    if state.lightcone:
        delta_lc_slices = delta_subset_lightcone(
            dj=dj,
            sim_out=part,
            state=state,
            scatter_gather_props=scatter_gather_props,
        )

        snapshot_data["delta_lc_slices"] = delta_lc_slices.as_dict()

    out = delta_subset(
        dj=dj,
        sim_output=part,
        state=state,
        scatter_gather_props=scatter_gather_props,
    )
    delta_slices, delta_full, ps = out
    snapshot_data["delta_slices"] = delta_slices.as_dict()

    def save_callback(_idx, callback_data):
        print(f"saving timestep {_idx}")
        if jax.process_index() != 0:
            return
        save_all_arrays_to_file(
            timestep_dir / f"step_{_idx}.asdf", callback_data, exploded=False
        )

    jax.debug.callback(save_callback, idx, snapshot_data)
