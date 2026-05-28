import dataclasses
from dataclasses import dataclass
from asyncio import run
from discodj_examples.simulation_run.config_state import ConfigState
from discodj_examples.simulation_run.device_setup import early_setup, device_setup

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--multicpu",
        action="store_true",
        default=False,
        help="run tests using multiple CPU acting as multiple devices",
    )


def pytest_configure(config):
    state = ConfigState(run_mode="singlecpu")

    if config.getoption("--multicpu"):
        state = dataclasses.replace(state, run_mode="cpu")
        early_setup(state)
        device_setup(state)

    # from discodj.multidevice.utils import get_mesh
    # import jax
    # jax.set_mesh(get_mesh())


def pytest_generate_tests(metafunc):
    if "runmode" in metafunc.fixturenames:
        if metafunc.config.getoption("--multicpu"):
            metafunc.parametrize("runmode", ["old_comm", "new_comm"])
        else:
            metafunc.parametrize("runmode", ["singlecpu"])


def pytest_collection_modifyitems(config, items):
    if config.getoption("--multicpu"):
        # user asked to run slow tests — do nothing
        return
    skip_multidevice = pytest.mark.skip(reason="multidevice test")
    for item in items:
        if "multidevice" in item.keywords:
            item.add_marker(skip_multidevice)
