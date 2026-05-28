"""
Things that need to be run before XLA is initialized (so before the first JAX operation)

While in theory all code should already be written in a way that doesn't execute JAX code on import,
it is still safer to run and import these functions before importing the majority of discodj and other own code.

These functions both apply XLA settings depending on the setup and normalize usage between multi-gpu, multi-cpu and single-device.

In theory I think early_setup and device_setup could be merged, I think in the past I was worried that the early_setup needs
to be executed even earlier.

"""
import os
import shutil
from pathlib import Path

import jax
from setproctitle import setproctitle

from .run_utils import nvidia_smi, add_xla_flag, get_xla_flags

from .config_state import ConfigState, updated_state


def early_setup(state: ConfigState):
    # add_xla_flag("--xla_disable_hlo_passes=constant_folding")
    if state.uses_job_system:
        job_based_id = state.job_id + "_" + state.job_process_index
        job_id = state.job_id
        restart_count = os.environ.get("SLURM_RESTART_COUNT", None)
        if restart_count is not None and int(restart_count) > 0:
            print(f"job has been restarted {restart_count} times")
            job_id = job_id + "_restart_" + restart_count
        print("job_id: ", job_id, job_based_id)
        state = updated_state(state, name=f"{state.run_mode}_{state.name}_{job_id}")
        nvidia_smi(expected_gpus=None)

        if state.dump_xla:
            try:
                job_idx = int(state.job_process_index)
            except ValueError:
                job_idx = 100
            if state.dump_xla_in_all_processes or job_idx == 0:
                add_xla_flag(f"--xla_dump_to=data/dump_host_{job_based_id}_{state.name}/")
        add_xla_flag("--xla_gpu_enable_latency_hiding_scheduler=true")
        add_xla_flag("--xla_dump_latency_hiding_schedule=true")

        add_xla_flag("--xla_gpu_enable_nccl_comm_splitting=true")
        add_xla_flag("--xla_gpu_enable_pipelined_all_gather=true")
        add_xla_flag("--xla_gpu_enable_pipelined_reduce_scatter=true")
        add_xla_flag("--xla_gpu_enable_pipelined_all_reduce=true")
        add_xla_flag("--xla_gpu_enable_pipelined_p2p=true")
        # proc_id=os.environ["SLURM_PROCID"]
        # add_xla_flag(f"--xla_gpu_pgle_profile_file_or_directory_path=data/jax_profiler/28468_{proc_id}/plugins/profile/2025_06_30_16_56_50/profile.pb")

    else:
        state = updated_state(state, name=f"{state.run_mode}_{state.name}")

        dump_name = state.name
        mpi_rank = os.environ.get("OMPI_COMM_WORLD_RANK", None)
        if mpi_rank is not None:
            state = updated_state(state, name=state.name + "_mpi")

            dump_name = state.name + f"_mpi_{mpi_rank}"
        if state.dump_xla:
            dump_dir = f"data/dump_{dump_name}"
            print("xla dump dir: ", dump_dir)
            # dump_dir = f"data/dump_host_cpu/"
            for file in Path(dump_dir).glob("*"):
                if file.is_dir():
                    dir = file
                    for file in dir.glob("*"):
                        file.unlink()
                    dir.rmdir()
                else:
                    file.unlink()
            add_xla_flag(f"--xla_dump_to={dump_dir}")
    return state


def device_setup(state: ConfigState, num_fake_devices=16):
    jax.config.update("jax_use_shardy_partitioner", True)

    mode = state.run_mode
    from jax._src import xla_bridge
    # def print_init_traceback(*args,**kwargs):
    #     print("JAX XLA was initialized here:")
    #     import traceback
    #     traceback.print_stack()
    #     exit()
    #
    # xla_bridge.get_backend=print_init_traceback

    # If this exception is raised, then JAX code was executed before calling this function which means the XLA settings
    # can no longer be set. To figure out the code to blame, uncomment the line before which prints a traceback at the
    # first time JAX is initialized.
    # see https://github.com/patrick-kidger/optimistix/pull/186 for an example
    assert xla_bridge._default_backend is None, "jax has been initialized before device_setup, see comments in device_setup.py for help"

    device_name = "cpu" if mode in ["cpu", "singlecpu"] else "gpu"
    single_device = False
    if mode == "cpu":
        ##### Running on CPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # deactivate GPU
        if os.getenv("OMPI_COMM_WORLD_RANK"):
            ##### Start with mpirun -np 4 python script.py
            ##### uses multi-process which is representative for JAX in multi-host setups
            ##### only uses MPI (and mpi4py) for the processes to find each other during initialize
            mode = "mpicpu"
            jax.distributed.initialize(cluster_detection_method="mpi4py")
            num_devices = len(jax.devices("cpu"))
        else:
            ##### create fake CPU devices in one JAX process
            ##### equivalent to one host with N GPUs
            num_devices = num_fake_devices
            add_xla_flag(f'--xla_force_host_platform_device_count={num_devices}')

    elif mode == "gpu":
        ##### Multi-GPU the way we use it on HPC clusters
        ##### Each jax process handles just one GPU
        uses_pbs = "PBS_JOBID" in os.environ
        nvidia_smi(expected_gpus=4)
        if not uses_pbs:
            jax.distributed.initialize(heartbeat_timeout_seconds=10, initialization_timeout=20)
            process_id = os.environ["SLURM_PROCID"]
            setproctitle(f"jax process {process_id}")
        else:
            jax.distributed.initialize(cluster_detection_method="mpi4py")

        num_devices = len(jax.devices("gpu"))
    elif mode == "gpu-one-process-per-node":
        nvidia_smi(expected_gpus=2)
        jax.distributed.initialize(local_device_ids=range(2))

        num_devices = len(jax.devices("gpu"))

    elif mode == "singlegpu":
        nvidia_smi(expected_gpus=1)
        num_devices = len(jax.devices("gpu"))
        single_device = True
        # update_global_options(single_device=True)
        assert num_devices == 1
    elif mode == "singlenode":
        nvidia_smi(expected_gpus=2)
        num_devices = len(jax.devices("gpu"))
        assert num_devices == 2
    elif mode == "singlecpu":
        num_devices = len(jax.devices("cpu"))
        single_device = True
        # update_global_options(single_device=True)
        assert num_devices == 1
    else:
        raise Exception(f"Unknown argument: {mode}")

    if single_device:
        print("single device mode; disabling multigpu related options")
        state = updated_state(
            state, 
            use_distributed_scatter_gather=False,
            padded_sim=False
        )

    devices = jax.devices()
    print("devices:", devices)
    return device_name, devices, mode, state


def device_closing(state: ConfigState, output_dir: Path = None, copy_slurm_output=True):
    """
    When submitting a slurm job and watching the output log, I find it incredibly useful to have a clear indicator that
    a program finished running. So I always call this function as the last thing, so it prints "finished" and the path to the output
    """
    dump_flag = get_xla_flags().get("--xla_dump_to", None)
    if dump_flag:
        print("xla dump:", dump_flag)
    if output_dir is not None:
        print("output directory:", output_dir)
    jax.distributed.shutdown()
    if state.uses_slurm and copy_slurm_output:
        # copy slurm output to output directory
        # This makes some assumptions on where they are stored,
        # but if they are not met they are just not copied
        output_dir_main = Path("data/output") / state.name
        proc_id = int(state.job_process_index)
        slurm_output_filename = f"slurm-{state.job_id}-{proc_id:02d}.out"
        output_name = Path("jobscripts/output") / slurm_output_filename
        if output_name.exists():
            shutil.copy(output_name, output_dir_main / "slurm-output.txt")
        else:
            output_name = Path("output") / f"slurm-{state.job_id}.out"
            if output_name.exists():
                shutil.copy(output_name, output_dir_main / "slurm-output.txt")
    (output_dir / "running").unlink()
    print("finished", state.name)
