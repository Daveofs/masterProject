from jztree.config import FofConfig
import dataclasses
from typing import TypeVar

import jax
import jax.numpy as jnp
import numpy as onp

from discodj.multidevice.communication import in_specs_from_particles, pad_particles
from discodj.multidevice.utils import get_mesh, is_multidevice
from discodj.nbody.nbody_dataclass import DJFofCatalogue, LightconeData, Particles

from .config_state import ConfigState
from .run_utils import compile_run_and_profile

PartT = TypeVar("PartT", Particles, LightconeData)


def get_fof_config(state:ConfigState):
    fof_cfg = FofConfig(
        alloc_fac_ilist=state.fof_alloc_fac_ilist,
        alloc_fac_distr_links=state.fof_alloc_fac_distr_links,
    )
    fof_cfg.tree.alloc_fac_nodes = state.fof_alloc_fac_nodes
    fof_cfg.catalogue.npart_min = state.fof_npart_min
    return fof_cfg

def calculate_fof(
    state: ConfigState, part: PartT, fof_cfg
) -> tuple[PartT, DJFofCatalogue]:
    from jztree.config import FofConfig
    from jztree.fof import FofCatalogue, fof_and_catalogue

    fof_cfg: FofConfig

    rlink = state.fof_link_length * state.boxsize / state.res

    part_specs = in_specs_from_particles(part)

    @jax.shard_map(
        out_specs=(part_specs, jax.P("gpus")), in_specs=(part_specs,), mesh=get_mesh()
    )
    def distr_fof(part: PartT) -> tuple[PartT, DJFofCatalogue]:
        if not part.padded:
            npart_per_device = part.npart_total // jax.device_count()
            part = pad_particles(
                part, onp.int32(npart_per_device * state.fof_pad_factor)
            )

        if jax.devices()[0].device_kind == "cpu":
            print("fake FoF halos on CPU")
            partf = part
            catalogue = FofCatalogue(
                jnp.array(20),
                count=jnp.concatenate([jnp.arange(20), jnp.zeros(80)]),
                com_pos=jnp.ones((100, 3))
            )
        else:
            partf, catalogue = fof_and_catalogue(
                part=part,
                rlink=rlink,
                boxsize=0 if state.lightcone else state.boxsize,
                cfg=fof_cfg
            )
        partf: PartT
        partf = dataclasses.replace(partf, num=jnp.atleast_1d(partf.num))
        catalogue = jax.tree.map(lambda a: jnp.atleast_1d(a), catalogue)

        catalogue_dj = DJFofCatalogue(
            **{
                f.name: getattr(catalogue, f.name)
                for f in dataclasses.fields(catalogue)
            }
        )

        return partf, catalogue_dj

    def local_fof(part: PartT) -> tuple[PartT, DJFofCatalogue]:
        if jax.devices()[0].device_kind == "cpu":
            print("fake FoF halos on CPU")
            partf = part
            catalogue = FofCatalogue(
                jnp.array(20),
                count=jnp.concatenate([jnp.arange(20), jnp.zeros(80)]),
                com_pos=jnp.ones((100, 3))
            )
        else:
            partf: PartT
            partf, catalogue = fof_and_catalogue(
                part=part,
                rlink=rlink,
                boxsize=0 if state.lightcone else state.boxsize,
                cfg=fof_cfg
            )
        catalogue_dj = DJFofCatalogue(
            **{
                f.name: getattr(catalogue, f.name)
                for f in dataclasses.fields(catalogue)
            }
        )

        return partf, catalogue_dj

    if is_multidevice():
        partf, catalogue = compile_run_and_profile(
            distr_fof,
            "distr_fof",
            {},
            part # (expanding_)shard_map doesn't allow kwargs
        )
    else:
        partf, catalogue = compile_run_and_profile(
            local_fof,
            "local_fof",
            {},
            part
        )
    return partf, catalogue
