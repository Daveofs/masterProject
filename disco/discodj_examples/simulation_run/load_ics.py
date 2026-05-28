import dataclasses
from enum import Enum
from pathlib import Path

import jax
import numpy as onp
from jax import Array
from jax import numpy as jnp

from discodj.cosmology.cosmology import Cosmology
from discodj.multidevice.communication import pad_particles, reorder_to_devices
from discodj.multidevice.utils import distribute_array_ax0, get_mesh, get_sharding_ax0
from discodj.nbody.nbody_dataclass import Particles, particle_id_to_3d
from discodj_examples.simulation_run.run_utils import global_timing_stats

from .config_state import ConfigState
from .run_utils import compile_run_and_profile


def compare_parameters(name: str, reference, provided):
    if provided is None:
        print(f"no parameter for {name} in config file, skipping check")
        return
    print(reference, provided, onp.allclose(reference, provided))
    if reference != provided and not onp.allclose(reference, provided):
        raise ValueError(
            f"Parameters for {name} do not match: {provided} should be {reference}"
        )


def load_local_subarray(data) -> Array:
    import h5py

    data: h5py.Dataset
    if jax.process_count() == 1:
        # only for local debugging
        # read all and reshard
        arr = jnp.array(data)
        arr: Array = jax.reshard(arr, get_sharding_ax0())
        return arr

    return distribute_array_ax0(data, get_mesh())


class ICFormat(Enum):
    pkdgrav = "pkdgrav"
    gadget4_monofonic = "gadget4_monofonic"


def load_ic_file(file: Path, state: ConfigState, cosmo: Cosmology) -> Particles:
    import h5py

    with h5py.File(file) as f:
        f: h5py.File
        # TODO: add header checks
        partgroup = f["PartType1"]
        header = dict(f["Header"].attrs)
        if "Units" in f.keys():
            units_header = dict(f["Units"].attrs)
        else:
            units_header = None

        if "PKDGRAV version" in header or ("Cosmology" in f and "dOmega0" in f["Cosmology"].attrs):
            format = ICFormat.pkdgrav
        elif units_header is None:
            format = ICFormat.gadget4_monofonic
        else:
            raise ValueError("Unknown IC format")

        h = header["HubbleParam"]
        boxsize = header["BoxSize"]
        if isinstance(boxsize, list):
            boxsize = boxsize[0]
        num_particles: int = partgroup["Coordinates"].shape[0]

        if format == ICFormat.pkdgrav:
            cosmology_header = dict(f["Cosmology"].attrs)

            redshift = header["Redshift"].squeeze()
            a = 1 / (redshift + 1)
            omega_b = cosmology_header["Omega_b"]
            omega_m = cosmology_header["Omega_m"]
            omega_lambda = cosmology_header["Omega_lambda"]
        elif format == ICFormat.gadget4_monofonic:
            a = header["Time"][0]
            omega_b = None
            omega_m = None
            omega_lambda = None

        expected_num_particles = state.res**3
        compare_parameters("Npart", expected_num_particles, num_particles)
        if not state.skip_ics_header_check:
            compare_parameters("boxsize", state.boxsize, boxsize)
            compare_parameters("a", state.a_ini, a)
            compare_parameters("h", cosmo.h, h)
            compare_parameters("omega_b", cosmo.Omega_b, omega_b)
            compare_parameters("omega_0", cosmo.Omega_m, omega_m)
            compare_parameters("omega_lambda", cosmo.Omega_de, omega_lambda)

        with global_timing_stats.timing("load data from IC file"):
            coords: Array = load_local_subarray(partgroup["Coordinates"])
            particle_ids = load_local_subarray(partgroup["ParticleIDs"])
            velocities = load_local_subarray(partgroup["Velocities"])

    # plt.hist(coords[:, 0], bins=50)
    # plt.show()
    coords = jax.reshard(coords, get_sharding_ax0())
    velocities = jax.reshard(velocities, get_sharding_ax0())
    particle_ids = jax.reshard(particle_ids, get_sharding_ax0())

    def initial_reorder(coords: Array, velocities: Array, particle_ids: Array):
        if format == ICFormat.pkdgrav:
            coords = (coords + 0.5) * boxsize
            v_factor = state.a_ini**2 / onp.sqrt(8 * onp.pi / 3) * boxsize
            velocities = velocities * v_factor

        elif format == ICFormat.gadget4_monofonic:
            coords = jnp.mod(coords, boxsize)
            velocities = velocities / (100 / a**1.5)

        npart_per_device = num_particles // jax.device_count()
        particle_ids_3d = particle_id_to_3d(particle_ids, state.res)
        part = Particles(
            psi=coords,  # this is fixed by subtracting q below
            vel=velocities,
            id_3d=particle_ids_3d,
            order="none",
            res=state.res,
            npart_total=num_particles,
            num=npart_per_device,
            boxsize=boxsize,
            res_pm=state.res_pm,
            a=0,
            fplus=0,
            vel_units="external",
        ).with_time(a, cosmo)

        assert state.padded_sim, "this reordering only works with padded simulations"

        part = dataclasses.replace(part, psi=part.psi - part.q())
        part = pad_particles(part, onp.int32(npart_per_device * state.fof_pad_factor))

        part = reorder_to_devices(part, set_worder_alignment=2)
        return part

    part = compile_run_and_profile(
        initial_reorder,
        "initial_reorder",
        coords=coords,
        velocities=velocities,
        particle_ids=particle_ids,
    )

    return part
