from discodj_examples.simulation_run.run_utils import compile_run_and_profile
from pathlib import Path

import jax
import numpy as onp
from jax_array_info import sharding_info
from nvtx import nvtx

from discodj import DiscoDJ
from discodj.multidevice.fft import irfftn_sharded_ax1_to_ax0
from discodj.multidevice.utils import get_sharding_ax0, is_multidevice, is_multiprocess
from discodj_native import rng_ngenic


def get_white_noise_field(dj: DiscoDJ, res: int, seed: int, use_cache: bool = True) -> jax.Array:
    if not is_multiprocess():
        white_noise_file = Path(f"data/white_noise_field_res_{res}_seed_{seed}.npy")

        print("noise")
        if white_noise_file.exists() and use_cache:
            with nvtx.annotate("loading whitenoise from disk"):
                print("loading")
                white_noise_field_real = onp.load(str(white_noise_file), mmap_mode="r")
        else:
            if res > 1024:
                raise RuntimeError("can't create large whf here")
            white_noise_field_real = dj.get_ngenic_noise(seed=seed).field
            if use_cache:
                onp.save(str(white_noise_file), white_noise_field_real)

        sharding_info(white_noise_field_real, "white_noise_field_real")
        white_noise_field_real = jax.device_put(white_noise_field_real, get_sharding_ax0() if is_multidevice() else None)
    else:
        white_noise_sharded_cache = Path(
            f"data/white_noise_field_{res}_{jax.process_index()}_{jax.device_count()}_{seed}.npy")
        if white_noise_sharded_cache.exists() and use_cache:
            white_noise_field_local = jax.device_put(onp.load(str(white_noise_sharded_cache), mmap_mode="r"))
        else:
            print("initialize ngenic")
            rng = rng_ngenic(seed, res)
            with nvtx.annotate("generating ngenic ICs"):
                print("generate local white noise field")
                white_noise_field_local = jax.device_put(rng.get_field_sharded(jax.device_count(), jax.process_index()))
                print("done")
            print("saving sharded white noise field to cache")
            if use_cache:
                onp.save(str(white_noise_sharded_cache), white_noise_field_local)
        white_noise_field_sharded = jax.make_array_from_single_device_arrays(
            (res, res, res // 2 + 1),
            get_sharding_ax0(),
            [white_noise_field_local])

        # TODO Hackathon:
        # This FFT is unfortunately less efficient than it could be as the distributed wnf in fourier space is sharded along
        # ax0 because of ngenic. So we need one extra reshard to get the real space wnf.
        # TODO Hackathon:
        # I guess in the future we could look into getting rid of the the two FFTs and use the fourier space wnf directly.

        # JIT distributed FFT as we are never in a jitted context here
        white_noise_field_real = compile_run_and_profile(
                irfftn_sharded_ax1_to_ax0,
                "ngenic_ics_fft",
                jit_arguments={"donate_argnums": [0]},
                x=white_noise_field_sharded * res ** (3 / 2)
            )
    return white_noise_field_real
