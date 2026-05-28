"""
# TODO Hackathon: I will keep this code contained within simulation_run and try to make it as little "global singleton" as possible
# and instead pass it as an argument

"""

import typing
from types import UnionType, NoneType
import argparse
import hashlib
import json
import os
import subprocess
import tomllib
from argparse import ArgumentParser
from dataclasses import dataclass, replace, fields
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ConfigState:
    name: str = "default"
    run_mode: str = "cpu"
    boxsize: float = 1000
    res: int = 64
    res_pm: int = 64
    n_order: int = 1
    a_ini: float = 0.02
    a_end: float = 1.0
    cosmo: str = "CamelsCV"
    precision: str = "single"
    numsteps: int = 8
    num_chunks: int = 8
    stepper: str = "bullfrog"
    time_var: str = "D"
    grad_kernel_order: int = 0
    laplace_kernel_order: int = 0
    worder: int = 2
    single_device: bool = False

    linear_ps_file: str | None = None
    ics_file: str | None = None
    skip_ics_header_check: bool = False

    padded_sim: bool = True

    gr_correction_res: int = 128
    gr_correction_file: str | None = None
    gr_correction_postprocessing: bool = False
    gr_correction_lightcone: bool = False

    double_precision_timetables: bool = True

    n_buffer_part_factor: int = 4
    n_buffer_part_fraction: int = 4
    n_buffer_grid_factor: int = 3
    n_buffer_grid_fraction: int = 4

    lightcone: bool = False
    # target radius of lightcone is lightcone_size_factor / lightcone_size_fraction * boxsize
    lightcone_size_factor: int = 1
    lightcone_size_fraction: int = 2
    lightcone_crossing_method: str = "florian"
    save_lightcone_data: bool = False
    rsd: bool = False
    healpix: bool = False
    healpix_nside: int = 128
    healpix_interpolation: bool = True

    # use_jax_profiler: bool = False
    use_distributed_scatter_gather: bool = True
    use_vjp_scatter: bool = False
    use_vjp_gather: bool = False
    scatter_gather_check: bool = False
    deconvolve_nbody: bool = False
    deconvolve_density_field: bool = False

    slice_axis: int = 1

    fNL_test: bool = False  # just a placeholder to test with fNL

    calculate_final_density_field: bool = True
    save_slice_at_every_timestep: bool = False
    calculate_fof: bool = False
    fof_link_length: float = 0.2
    fof_pad_factor: float = 0.5
    fof_alloc_fac_nodes: float = 1.0
    fof_alloc_fac_ilist: int = 32
    fof_alloc_fac_distr_links: float = 0.01
    fof_npart_min: int = 20
    fof_stats: bool = False

    heft: bool = False
    heft_bias_order: int = 1

    # kill and requeue the job if it hasn't passed the first trivial operation after N seconds
    # set to 0 to disable
    timeout_for_first_sync_point: float = 120  # seconds

    dump_xla: bool = True
    dump_xla_in_all_processes: bool = False
    save_final_field: bool = False
    save_hdf5_snapshot: bool = False
    snapshot_every_timestep: bool = False
    hdf_format: str = "unmodified"
    quiet_compile: bool = False
    never_plt_show: bool = False

    powerspectrum: bool = False

    # RNG seeds
    seed_ngenic: int = 180723
    seed_jaxrng: int = 12345

    jax_rng_ics: bool = False

    locked: bool = False

    def __post_init__(self):
        assert self.run_mode in {
            "cpu",
            "gpu",
            "gpu-one-process-per-node",
            "singlegpu",
            "singlenode",
            "singlecpu",
        }
        assert self.precision in {"single", "double"}
        assert self.stepper in {"bullfrog", "fastpm", "symplectic"}
        assert self.slice_axis in {0, 1, 2}
        assert self.time_var == "D"  # improve later
        if self.rsd:
            assert self.lightcone, "need lightcone for RSD"
        if self.gr_correction_lightcone or self.gr_correction_postprocessing:
            assert self.gr_correction_file is not None, "need GR correction table"
            assert self.gr_correction_lightcone or self.gr_correction_postprocessing
        if self.gr_correction_lightcone or self.save_lightcone_data:
            assert self.lightcone
        if self.gr_correction_file is not None:
            assert self.gr_correction_file.endswith(".asdf")
        if self.healpix:
            assert self.lightcone
        if self.lightcone:
            assert self.numsteps > 1
        if self.padded_sim:
            if self.run_mode not in ["cpu", "gpu"]:
                print("Warning: padded sim is not needed in single-device setups")
        assert self.numsteps >= 0

    def as_dict(self):
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["n_buffer_part"] = self.n_buffer_part
        data["n_buffer_grid"] = self.n_buffer_grid
        data["chunk_size"] = self.chunk_size
        data["num_devices"] = self.num_devices
        data["git_hash"] = self.git_hash
        data["jz_tree_version"] = self.jz_tree_version
        data["uses_slurm"] = self.uses_slurm
        data["uses_pbs"] = self.uses_pbs
        data["job_id"] = self.job_id
        data["job_process_index"] = self.job_process_index
        data.pop("locked")
        return data

    def options_hash(self, extra_data: dict = None, extra_exclude: list[str] = None):
        options = self.as_dict()
        if extra_exclude is None:
            extra_exclude = []
        for opt in [
            "dump_xla",
            "save_all_arrays",
            "save_hdf5_snapshot",
        ] + extra_exclude:
            options.pop(opt)
        if extra_data is not None:
            options.update(extra_data)
        opt_str = json.dumps(options, sort_keys=True)
        return hashlib.sha256(opt_str.encode()).hexdigest()

    @property
    def num_devices(self):
        import jax
        from jax._src import xla_bridge

        if xla_bridge._default_backend is None:
            # don't accidentally initialize XLA here
            return -1
        return jax.device_count()

    @property
    def n_buffer_part(self):
        return (
            self.n_buffer_part_factor
            * self.res
            // self.num_devices
            // self.n_buffer_part_fraction
        )

    @property
    def n_buffer_grid(self):
        return (
            self.n_buffer_grid_factor
            * self.res_pm
            // self.num_devices
            // self.n_buffer_grid_fraction
        )

    @property
    def N(self):
        return self.res

    @property
    def chunk_size(self):
        return self.res**3 // self.num_chunks

    @property
    def git_hash(self):
        try:
            out = subprocess.run(
                ["git", "describe", "--always", "--dirty"],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            return None
        return out.stdout.decode().strip()

    @property
    def jz_tree_version(self):
        try:
            import jztree._version

            return jztree._version.version
        except ModuleNotFoundError:
            return None

    @property
    def uses_slurm(self):
        return "SLURM_JOB_ID" in os.environ

    @property
    def uses_pbs(self):
        return "PBS_JOBID" in os.environ

    @property
    def uses_job_system(self):
        return self.uses_slurm or self.uses_pbs

    @property
    def job_id(self):
        if self.uses_slurm:
            return os.environ["SLURM_JOB_ID"]
        elif self.uses_pbs:
            return os.environ["PBS_JOBID"]
        return "job_id"

    @property
    def job_process_index(self):
        if self.uses_slurm:
            return os.environ["SLURM_PROCID"]
        elif self.uses_pbs:
            try:
                return os.environ["PBS_JOBID"]
            except KeyError:
                return "unknown"
        return "unknown"

    def __str__(self):
        output = "global state:\n"
        for k, v in self.as_dict().items():
            output += f"{k:<20} = {v}\n"
        return output

    def changes(self, default: "ConfigState" = None):
        if default is None:
            default = ConfigState()
        own_values = self.as_dict()
        default_values = default.as_dict()
        changed_values = {}
        for k, v in own_values.items():
            if k in ["name", "n_buffer_part", "n_buffer_grid", "chunk_size"]:
                continue
            default_val = default_values[k]
            if v != default_val:
                changed_values[k] = v
        return changed_values


def get_global_state_from_config(config: Path) -> ConfigState:
    with config.open("rb") as f:
        data = tomllib.load(f)

    def get_path_or_none(data, attr):
        if attr not in data:
            return None
        if data[attr] is None:
            return None
        return Path(data[attr])

    # data["ref_file"] = get_path_or_none(data, "ref_file")
    # data["linear_ps_file"] = get_path_or_none(data, "linear_ps_file")

    return ConfigState(**data)


def get_global_option_parser():
    parser = ArgumentParser()
    parser.add_argument("config", type=Path, nargs="?")
    for f in fields(ConfigState):
        name = f"--{f.name.replace('_', '-')}"
        kwargs = {}
        if isinstance(f.type,UnionType):
            kwargs["type"]=[t for t in typing.get_args(f.type) if t is not NoneType][0]
        elif f.type is bool:
            kwargs["action"] = "store_true"
        else:
            kwargs["type"] = f.type
        # kwargs["default"] = f.default
        kwargs["default"] = argparse.SUPPRESS
        kwargs["dest"] = f.name

        parser.add_argument(name, **kwargs)
        if f.type is bool:
            kwargs["action"] = "store_false"
            name = name.replace("--", "--no-")
            parser.add_argument(name, **kwargs)
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--z-ini", dest="z_ini", type=float)
    parser.add_argument("--export-config", action="store_true")
    return parser


def get_global_state_from_args():
    parser = get_global_option_parser()
    args = vars(parser.parse_args())
    config = args.pop("config")
    if config is not None:
        state = get_global_state_from_config(config)
    else:
        state = ConfigState()
    double = args.pop("double")
    if double:
        args["precision"] = "double"
    zini = args.pop("z_ini")
    if zini is not None:
        aini = 1 / (zini + 1)
        args["a_ini"] = aini
        print(f"z_ini = {zini}")
        print(f"setting a_ini = {aini}")
    export_config = args.pop("export_config")
    state_new = replace(state, **args)
    if export_config is not None and export_config:
        changes = state_new.changes()
        import tomli_w

        changes_toml = tomli_w.dumps(changes)
        print("\nconfig.toml:")
        print(changes_toml)
        exit()

    return state_new


def updated_state(state: ConfigState, **kwargs) -> ConfigState:
    return replace(state, **kwargs)


####
# Everything below is the global-singleton version of this object that I will avoid using
# I just keep this code for reference in case it becomes useful again for another test

"""
This is an ugly hack and should be removed in the final version.

But having one global object that holds state like
"should this option be used" allows comparing different variants of the code
without always having to refactor all discodj functions to pass along the new options
(that mostly likely are only temporarily for debugging)

This requires that the global state object is only modified before starting the simulation
and is read-only afterwards.

This way any place in discodj can do a
```
from discodj.core.global_state import gs
if gs.some_option:
    ...
```
"""

_global_state: ConfigState = ConfigState()


def set_global_state(global_state: ConfigState):
    global _global_state
    if _global_state.locked:
        raise RuntimeError(
            "The global options can't be updated as they have been locked"
        )
    _global_state = global_state


def set_global_state_from_args():
    state = get_global_state_from_args()
    set_global_state(state)


def update_global_options(**options):
    global _global_state
    if _global_state.locked:
        raise RuntimeError(
            "The global options can't be updated as they have been locked"
        )
    _global_state = replace(_global_state, **options)
    return _global_state


def lock_global_options():
    global _global_state
    _global_state = replace(_global_state, locked=True)


def set_global_state_from_config(config: Path):
    options = get_global_state_from_config(config)
    set_global_state(options)


def get_global_state() -> ConfigState:
    return _global_state


class _GlobalStateProxy:
    def __getattr__(self, name: str):
        return getattr(_global_state, name)

    def __repr__(self):
        return repr(_global_state)

    def __str__(self):
        return str(_global_state)


gs: ConfigState = _GlobalStateProxy()
