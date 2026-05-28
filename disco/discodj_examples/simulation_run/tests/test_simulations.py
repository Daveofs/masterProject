import dataclasses
from contextlib import chdir
from pathlib import Path

import jax
from numpy.testing import assert_allclose
from reference_data import default_sim

from discodj.core.array_file_utils import load_arrays_from_file
from discodj_examples.simulation_run.config_state import ConfigState
from discodj_examples.simulation_run.simulation_run import simulation_run


def run_end_to_end_simulation(state: ConfigState, dir: Path):
    with chdir(dir):
        (dir / "data").mkdir(exist_ok=True)
        device_name = "cpu"
        devices = jax.devices()
        mode = "singlecpu"
        data = device_name, devices, mode, state
        out_dir = simulation_run(data)
        (dir / "output").symlink_to(out_dir)
    return dir / out_dir


def load_simulation(out_dir: Path):
    data = load_arrays_from_file(
        out_dir / "arrays/save_data.asdf", force_unsharded=True
    )
    return data


DATA_DICT = dict[str, jax.Array]


def recursive_compare(ref: DATA_DICT, test: DATA_DICT):
    for k, v in ref.items():
        if isinstance(v, dict):
            recursive_compare(ref[k], test[k])
        else:
            assert_allclose(ref[k], test[k], rtol=1e-5, atol=1e-5)


def apply_runmode(config: ConfigState, runmode):
    if runmode == "old_comm":
        config = dataclasses.replace(config, run_mode="cpu")
    elif runmode == "new_comm":
        config = dataclasses.replace(config, run_mode="cpu", padded_sim=True)
    elif runmode == "singlecpu":
        config = dataclasses.replace(config, run_mode="singlecpu")
    return config


def test_default_sim(tmp_path, runmode):
    config = ConfigState(never_plt_show=True, powerspectrum=True)
    config: ConfigState = apply_runmode(config, runmode)

    out_dir = run_end_to_end_simulation(config, tmp_path)
    print(out_dir)
    af, data = load_simulation(out_dir)
    ps = data["power_spectrum"]
    print(ps,default_sim)
    recursive_compare(ps,default_sim)

    af.close()
