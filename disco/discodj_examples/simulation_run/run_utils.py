"""
All other utility functions that can't be grouped in other ways, but are very helpful in running and debugging simulations.

Feel free to merge functions from here into DISCO-DJ proper.
"""

import contextlib
import json
import os
import socket
import subprocess
import threading
import time
import typing_extensions
from datetime import datetime
from pathlib import Path
from sys import argv
from typing import Callable, ParamSpec, TypeVar

import jax
import nvtx
from jax._src.stages import Lowered, Compiled
from jax_array_info.utils import pretty_byte_size
from jaxlib._jax import hlo_module_to_dot_graph, hlo_module_from_text, CompiledMemoryStats

from .config_state import ConfigState

P = ParamSpec("P")
R = TypeVar("R")


def compile_run_and_profile(func: Callable[P, R], name: str, jit_arguments: dict | None = None, *args: P.args,
                            **kwargs: P.kwargs) -> R:
    """
    For the main simulation_run it is useful to break up the simulation into major jitted blocks
    For each of them we are interested in both their compile time,
    some stats based on the compilation (e.g. memory usage)
    and the runtime.

    This wrapper function handles all of that while keeping the code in simulation_run as short as possible
    """
    if jit_arguments is None:
        jit_arguments = {}
    with global_timing_stats.timing(f"compiling {name}"):
        func = jax.jit(func, **jit_arguments)
        func_low = func.lower(*args, **kwargs)
        func_comp = func_low.compile()
    mem_stats = memory_analysis(func_comp, name, low=func_low)
    global_compilation_stats.add(name, mem_stats, hlo=func_comp.as_text())

    with global_timing_stats.timing(f"running {name}"):
        output = func(*args, **kwargs)
        # TODO: I am ambivalent about if this should be here or not
        # Without it all timing statements like "N-Body takes Xs" are rather meaningless
        # But without it, JAX can also move on to the next thing faster
        # and e.g. compile the next function
        # while the GPU has not yet finished calculating the results
        return jax.block_until_ready(output)


def nvidia_smi(expected_gpus: int | None = 4):
    """
    Run (and print) nvidia_smi and make sure the expected number of GPUs are detected
    """
    print(socket.gethostname())
    out = subprocess.run(["nvidia-smi"], capture_output=True)
    text = out.stdout.decode()
    print(text)
    assert out.returncode == 0
    num_local_gpus = text.count("NVIDIA") - 1
    if expected_gpus is not None and num_local_gpus != expected_gpus:
        raise RuntimeError(f"missing GPUs on {socket.gethostname()}: expected {expected_gpus}, got {num_local_gpus}")


def todotgraph(x):
    return f"// {datetime.now()}\n" + hlo_module_to_dot_graph(hlo_module_from_text(x))


def debug_graph_output(lowered_function: Lowered, name: str, run_graphviz: bool = False):
    platform = jax.local_devices()[0].platform
    compiled = lowered_function.compile()
    print(f"DEBUG cost analysis for {name}")
    memory_analysis(compiled, name, lowered_function)
    hlo = compiled.as_text()

    if jax.process_index() == 0:
        dotfile = Path(f"{name}_{platform}.dot")
        dotfile.write_text(todotgraph(hlo))
        if run_graphviz:
            with dotfile.with_suffix(".svg").open("w") as f:
                subprocess.run(["dot", "-Tsvg", str(dotfile)], check=True, stdout=f)
        Path(f"{name}_{platform}.txt").write_text(hlo)


def memory_analysis(comp: Compiled, name: str, low=None):
    stats: CompiledMemoryStats = comp.memory_analysis()
    if stats is None:
        return
    stats_dict = {
        "generated_code": stats.generated_code_size_in_bytes,
        "temp": stats.temp_size_in_bytes,
        "argument": stats.argument_size_in_bytes,
        "output": stats.output_size_in_bytes,
        "alias": stats.alias_size_in_bytes,
        "peak": stats.peak_memory_in_bytes,
        "host_temp": stats.host_temp_size_in_bytes,
    }
    if low:
        try:
            from pytest_jax_bench.utils import folded_constants_bytes
            stats_dict["const"] = folded_constants_bytes(low)
        except Exception as e:
            print("calculating folded constant size failed")
            stats_dict["const"] = None

    try:
        import rich
        from rich.table import Table
    except ImportError:
        print("could not import rich, printing raw data instead of table")
        print(f"Memory Stats '{name}'")
        print(stats_dict)
        return stats_dict

    table = Table(title=f"Memory Stats '{name}'", title_style="red")
    table.add_column("name")
    table.add_column("size", justify="right")
    table.add_column("size (raw)", justify="right")

    for k, v in stats_dict.items():
        if v is None:
            continue
        table.add_row(
            k,
            pretty_byte_size(v),
            str(v),
        )
    console = rich.console.Console()
    console.print(table)
    return stats_dict


def get_xla_flags():
    """
    parse existing XLA flags into key-value dict

    boolean flags need an explicit --flagname=true
    """
    xla_flags = {}
    if "XLA_FLAGS" in os.environ:
        existing_flags = os.environ["XLA_FLAGS"].split()
        for existing_flag in existing_flags:
            k, v = existing_flag.split("=")
            xla_flags[k] = v
    return xla_flags


def add_xla_flag(flag: str) -> None:
    """
    Modify XLA_FLAGS to add a new XLA flag to the existing list

    boolean flags need an explicit --flagname=true
    """

    xla_flags = get_xla_flags()

    key, value = flag.split("=")
    if key in xla_flags and xla_flags[key] != value:
        print(f"overwriting existing XLA_FLAGS entry {key}: {xla_flags[key]} -> {value}")
    xla_flags[key] = value

    os.environ["XLA_FLAGS"] = " ".join(f"{k}={v}" for k, v in xla_flags.items())


def init_output_dir(state: ConfigState) -> tuple[Path, Path]:
    """
    create output directory for simulation and fill it with basic metadata.

    Returns output directory this rank should use and the main output directory that rank 0 is using.

    Creates this folder structure in data/output/simname
    .
    ├── meta.json
    ├── argv.txt
    ├── other_ranks
    │   ├── 1
    │   │   ├── meta.json
    │   │   └── argv.txt
    ...

    """
    out = Path("data/output")
    out.mkdir(exist_ok=True)
    output_dir = out / state.name
    output_dir.mkdir(exist_ok=True)
    output_dir_main = output_dir
    if jax.process_index() > 0:
        output_dir = output_dir / "other_ranks" / str(jax.process_index())
        output_dir.mkdir(exist_ok=True, parents=True)
    with (output_dir / "meta.json").open("w") as f:
        json.dump(state.as_dict(), f, ensure_ascii=False, indent=4)
    with (output_dir / "argv.txt").open("w") as f:
        argv_line = " ".join(argv)
        print(argv_line)
        f.write(argv_line + "\n")
    (output_dir / "running").touch()
    return output_dir, output_dir_main


@typing_extensions.deprecated("untested")
def summarize_leaf(x):
    if isinstance(x, jax.numpy.ndarray):
        return f"<Array shape={x.shape}, dtype={x.dtype}>"
    return x  # leave other types alone


@typing_extensions.deprecated("untested")
def summarize_pytree(pytree):
    from black import format_str, FileMode
    out = str(jax.tree.map(summarize_leaf, pytree))
    return format_str(out, mode=FileMode())


@contextlib.contextmanager
def suppress_stdout(state: ConfigState):
    """
    depending on the number of print() inside a jitted function, tracing it can generate a lot of output
    that is useful for debugging, but can get annoying when running simulations.

    This is a surprisingly useful context manager to hide all output while jit-compiling.
    I often use it like this:
    >>> global_timing_stats.start_timing("compiling some_function")
    >>> with suppress_stdout():
    >>>     some_function_low = some_function.lower(input)
    >>>     some_function_comp = some_function_low.compile()
    >>> global_timing_stats.stop_timing("compiling some_function")
    >>> memory_analysis(some_function_comp, "some_function", low=some_function_low)
    >>>
    >>> global_timing_stats.start_timing("running some_function")
    >>> output = some_function(input)
    >>> global_timing_stats.stop_timing("running some_function")

    # TODO: Variant that redirects to logfile instead of hiding completely
    """
    if not state.quiet_compile:
        yield
        return
    with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull):
        yield


class JobWatcher:
    def __init__(self, timeout: float):
        self.enabled = False
        if jax.process_index() != 0 or timeout == 0:
            return
        self.slurm_id = os.environ.get("SLURM_JOB_ID", None)
        if self.slurm_id is None:
            return
        self.timeout = timeout
        self._reached = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        self.enabled = True

    def reached(self):
        if not self.enabled:
            return
        self._reached.set()

    def _watch(self):
        if not self._reached.wait(self.timeout):
            print("waiting too long")
            print("jobID is ", self.slurm_id)
            restart_count_str = os.environ.get("SLURM_RESTART_COUNT", None)
            print("restart count: ", restart_count_str)

            if restart_count_str is not None and int(restart_count_str) > 5:
                print("too many restarts")
                return
            print("requeuing job")
            subprocess.run(["scontrol", "requeue", str(self.slurm_id)], check=True)
            print("job requeued, it will be killed soon")


def profiler_to_data_for_profile_guided_latency_estimator(using_slurm:bool,use_jax_profiler:bool, profile_dir:Path):
    """
    based on https://docs.jax.dev/en/latest/gpu_performance_tips.html#manual-pgle
    no longer used
    """
    if not using_slurm or not use_jax_profiler:
        return
    # profile_dir = Path(f"data/jax_profiler/{state.name}")
    directories = profile_dir.glob('plugins/profile/*/')
    directories = [d for d in directories if d.is_dir()]
    rundir = directories[-1]
    print("rundir:", rundir)

    from jax.experimental import profiler as exp_profiler

    fdo_profile = exp_profiler.get_profiled_instructions_proto(os.fspath(rundir))

    dump_profile = rundir / f'profile_{jax.process_index()}.pb'
    dump_profile.parent.mkdir(parents=True, exist_ok=True)
    dump_profile.write_bytes(fdo_profile)
    print(f"written {dump_profile}")

class GlobalTimingStats:
    """
    simple global singleton to recording timings of different actions
    Also adds nvtx ranges, so that timings also show up in the nvidia profiler

    The _callback based versions don't seem to be working correctly.

    global_timing_stats is a per-process global singleton that can be used from everywhere.
    """

    def __init__(self):
        self.recorded = []
        self.general_meta = {}
        self.start_times = {}
        self.nvtx_ranges = {}

    def add_timing(self, type: str, timing: float, meta: dict):
        additional_meta = {
            "slurm_id": os.environ.get("SLURM_JOB_ID", None),
            "num_devices": jax.device_count(),
            "process_index": jax.process_index(),
            "time": datetime.now().isoformat(),
        }
        self.recorded.append({
            "timing": timing,
            "type": type,
            "meta": {**additional_meta, **meta},
        })

    def start_timing(self, type: str, *args):
        self.nvtx_ranges[type] = nvtx.start_range(message=type, domain="custom timing")
        self.start_times[type] = time.perf_counter()
        print(f"start {type}")

    @typing_extensions.deprecated("Don't trust the output of this method")
    def start_timing_callback(self, type: str, *args):
        jax.debug.callback(lambda *args: self.start_timing(type, *args), *args)

    def stop_timing(self, type: str, meta: dict = None, *args):
        if meta is None:
            meta = {}
        nvtx.end_range(self.nvtx_ranges[type])
        end_time = time.perf_counter()
        timing = end_time - self.start_times[type]
        if timing < 1:
            pretty_timing = f"{timing * 1000:.1f} ms"
        else:
            pretty_timing = f"{timing:.2f} s"
        print(f"{type}: {pretty_timing} {meta or ''}")
        self.add_timing(type, timing, meta)
        return timing

    @typing_extensions.deprecated("Don't trust the output of this method")
    def stop_timing_callback(self, type: str, meta: dict, *args):
        def clean_value(val):
            if isinstance(val, jax.Array):
                return val.item()

        def callback(meta, *args):
            meta = {k: clean_value(v) for k, v in meta.items()}
            self.stop_timing(type, meta)

        jax.debug.callback(callback, meta, *args)

    def add_general_meta(self, meta: dict):
        self.general_meta = meta

    def save_stats(self, file: Path):
        with file.open("w") as f:
            json.dump({"timings": self.recorded, "general_meta": self.general_meta}, f, indent=4, ensure_ascii=False)

    @contextlib.contextmanager
    def timing(self, type: str, meta: dict = None, *args):
        self.start_timing(type, *args)
        try:
            yield
        finally:
            self.stop_timing(type, meta=meta, *args)

global_timing_stats = GlobalTimingStats()


class GlobalCompilationStats:
    """
    conceptually similar to GlobalTimingStats, but storing memory stats and HLO
    """

    def __init__(self):
        self.memory_stats: dict[str, dict[str, int | float]] = {}
        self.hlo: dict[str, str] = {}

    def add(self, name: str, memory_stats: dict[str, int | float], hlo: str):
        self.memory_stats[name] = memory_stats
        self.hlo[name] = hlo

    def save_stats(self, directory: Path):
        with (directory / "memory_stats.json").open("w") as f:
            json.dump(self.memory_stats, f, indent=4, ensure_ascii=False)
        hlo_directory = directory / "hlo"
        hlo_directory.mkdir(parents=True, exist_ok=True)
        for name, hlo in self.hlo.items():
            hlo_file = hlo_directory / f"{name}.txt"
            hlo_file.write_text(hlo)


global_compilation_stats = GlobalCompilationStats()
