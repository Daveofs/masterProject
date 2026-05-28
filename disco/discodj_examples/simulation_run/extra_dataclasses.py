from dataclasses import dataclass
from dataclasses import field as dc_field

import jax


@jax.tree_util.register_dataclass
@dataclass
class PowerSpectrum:
    k: jax.Array
    P: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class FieldSlices:
    mean: jax.Array
    slice: jax.Array
    thin_slice: jax.Array
    slice_layer: int = dc_field(metadata=dict(static=True))
    slice_range: tuple[int, int] = dc_field(metadata=dict(static=True))
    slice_thickness: int = dc_field(metadata=dict(static=True))

    def as_dict(self):
        return self.__dict__
