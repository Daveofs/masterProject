import dataclasses

from jax import numpy as jnp

from discodj.cosmology.cosmology import Cosmology
from .config_state import ConfigState


def cosmology_from_config(state: ConfigState) -> Cosmology:
    # TODO: allow arbitray cosmology input
    cosmo = Cosmology.from_string_or_dict(state.cosmo, 32 if state.precision == "single" else 64,
                                          jnp.float64 if state.precision == "double" else jnp.float32)

    if state.fNL_test:
        cosmo = dataclasses.replace(cosmo, fNL_loc=0)
    return cosmo
