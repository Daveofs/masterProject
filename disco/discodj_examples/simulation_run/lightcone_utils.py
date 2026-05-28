"""
These are some helper functions to handle lightcone specific output
Most of this is based on Alejandro's code.

I guess once most of this is more stable, it should instead become part of DISCO-DJ
"""

from discodj.core.utils import tree_map_by_len
from discodj_examples.simulation_run.run_utils import compile_run_and_profile
import jax
import jax_healpy
import numpy as onp
from jax import numpy as jnp
from matplotlib import pyplot as plt

from discodj import DiscoDJ
from discodj.core.constants import constants
from discodj.multidevice.communication import compact_sim
from discodj.multidevice.utils import get_sharding_none, is_multidevice
from discodj.nbody.nbody_dataclass import LightconeData, Particles

from .config_state import ConfigState


def lightcone_output(S: jax.Array, obs:jax.Array, r_lc_ini:float, dj: DiscoDJ, do_rsd: bool):
    """
    Particles that are not in the lightcone will have a position of 0
    and therefore be True in is_in_lightcone.

    Using this as the weight in the scatter makes sure particles outside the lightcone will not contribute
    to the density.

    But with multi-gpu this is not enough as those particles will still be considered to be in (0,0,0)
    and therefore occupy space in the communication buffers.
    So to avoid them being seen "outside" their local range in communicate_particle_buffers I will also
    set them to nan.
    """
    is_in_lightcone = S[:, 0] != 0.
    # S = (z+1, X,Y,Z, p_rad)
    POS = S[:, 1:4]
    dist = jnp.linalg.norm(POS, axis=-1)
    inside_expected_lc_size = dist <= r_lc_ini
    is_in_lightcone = is_in_lightcone & inside_expected_lc_size
    # POS = jnp.where(is_imn_lightcone[:, None], POS, jnp.nan)
    if do_rsd:
        thomas_original_implementation = False
        if thomas_original_implementation:
            zp1_sim = S[:, :1]
            a_sim = 1 / zp1_sim  # this devides by zero for non-crossed particles
            a_sim = jnp.nan_to_num(a_sim, nan=1.)  # so we set the nan to something.
            Sxyz_sim = S[:, 1:-1] + 1e-4 * (
                ~is_in_lightcone[..., None])  # to prevent division by zero norm vector in the next line
            norm_sim = Sxyz_sim / jnp.linalg.norm(Sxyz_sim, axis=-1, keepdims=True)
            V_los = S[:, -1:]
            V_lc = V_los * norm_sim
            prefac = 1 / dj.cosmo.E(a_sim) / a_sim ** 2.
            POS_rsd = Sxyz_sim + V_lc * jax.lax.stop_gradient(prefac)
        else:
            z_values_plus_one = S[:, 0]
            eps = 1e-12
            z_values_plus_one_save = jnp.maximum(z_values_plus_one, eps)
            a_values = 1 / z_values_plus_one_save

            sumsq = jnp.sum(POS * POS, axis=-1, keepdims=True)
            norm = jnp.sqrt(jnp.maximum(sumsq, eps))
            norm_sim = POS / norm

            V_los = S[:, -1]
            V_lc = V_los[:, None] * norm_sim
            prefac = 1 / dj.cosmo.E(a_values) / a_values ** 2.

            POS_rsd = POS + V_lc * prefac[:, None] * is_in_lightcone[:, None]

        POS = POS_rsd + obs
    else:
        POS = POS + obs

    # POS = jnp.reshape(POS, (dj.res, dj.res, dj.res, 3))

    return POS, is_in_lightcone



def lightcone_output_dataclass(
    state: ConfigState,
    dj: DiscoDJ,
    sim_out: Particles,
    compact_array: bool,
    with_extra_data: bool = False,
) -> tuple[LightconeData, int | jax.Array]:
    obs = jnp.array([dj.boxsize / 2, dj.boxsize / 2, dj.boxsize / 2])
    r_lc_ini = state.boxsize * state.lightcone_size_factor / state.lightcone_size_fraction

    POS, is_in_lightcone = lightcone_output(
        dj=dj,
        S=sim_out.s,
        obs=obs,
        r_lc_ini=r_lc_ini,
        do_rsd=state.rsd
    )
    POS = POS.reshape(-1, 3)
    POS = jnp.where(is_in_lightcone[..., None], POS, jnp.nan)
    assert sim_out.mass is not None
    lc_data = LightconeData(
        pos=POS,
        obs=obs,
        mass=sim_out.mass,
        num=sim_out.num, # TODO: this is not really correct and gets replaced with the correct number per GPU in compact_sim below
        npart_total=sim_out.npart_total,
        padded=sim_out.padded,
        worder_alignment=sim_out.worder_alignment,
        boxsize=sim_out.boxsize,
        res=sim_out.res,
        res_pm=sim_out.res_pm
    )
    if with_extra_data:
        lc_data.zplus1 = sim_out.s[:, 0]
        lc_data.v_rad = sim_out.s[:, -1]
    if compact_array:
        # make sure the nan particles are in a continous block at the end
        lc_data = compact_sim(lc_data, is_in_lightcone)
        with jax.enable_x64():
            num_part_total = jnp.sum(lc_data.num)
    else:
        with jax.enable_x64():
            num_part_total = jnp.sum(~jnp.isnan(lc_data.pos[:, 0]))

    return lc_data, num_part_total


def lightcone_data_combined(particles: Particles, state: ConfigState, dj: DiscoDJ, save_data, compact_array:bool):
    """
    lightcone interface for simulation_run
    """
    lightcone_data, num_particles = compile_run_and_profile(
        lightcone_output_dataclass,
        "lightcone_output_dataclass",
        jit_arguments={"static_argnames": ["state", "dj", "compact_array", "with_extra_data"]},
        state=state,
        dj=dj,
        sim_out=particles,
        compact_array=compact_array,
        with_extra_data=True
    )
    if is_multidevice():
        lightcone_data.npart_total = int(onp.ceil(num_particles / jax.device_count()) * jax.device_count())
        save_data["lightcone_data.npart_total"] = lightcone_data.npart_total
        lightcone_data.npart_total_type = "rounded_up_to_devices"
    else:
        # because we are outside of JIT and on a single device here
        # we can cut the lightcone particle array to the actual lenght
        # of lightcone particles
        lightcone_data = tree_map_by_len(
            lambda x: x[: lightcone_data.num[0]],
            lightcone_data,
            N=lightcone_data.pos.shape[0],
        )
        lightcone_data.npart_total = int(lightcone_data.num[0])

    if False:
        print(lightcone_data.num.addressable_data(0))
        local_subset=lightcone_data.pos.addressable_data(0)[0][:lightcone_data.num.addressable_data(0)[0]]
        print(lightcone_data.pos.addressable_data(0).shape)
        print(lightcone_data.num.addressable_data(0))
        print("nans",jax.numpy.isnan(local_subset).sum())
        exit()

    return lightcone_data


def lightcone_circle(state:ConfigState, ax: plt.Axes) -> None:
    lc_size = state.lightcone_size_factor / state.lightcone_size_fraction * state.boxsize
    theta = onp.linspace(-1 * onp.pi, 1 * onp.pi, 100)
    y_val = lc_size * onp.sin(theta)  # + obs[1]
    z_val = lc_size * onp.cos(theta)  # + obs[2]
    ax.plot(y_val, z_val, c='r', linestyle='dashed', linewidth=.9)


def boxsize(state:ConfigState, dj: DiscoDJ):
    c = constants.c * dj.cosmo.h
    eta_obs = dj.cosmo.a_to_conformalt(1.)
    print("eta_obs -> ", eta_obs)

    a_ini = state.a_ini
    a_end = 1.

    eta_ini = dj.cosmo.a_to_conformalt(a_ini)
    eta_fin = dj.cosmo.a_to_conformalt(a_end)

    # Boxsize/2
    r_ini = c * jnp.abs(eta_ini - eta_obs)
    r_fin = c * jnp.abs(eta_fin - eta_obs)

    print('\na_ini -> ', a_ini)
    print('z_ini -> ', 1 / a_ini - 1.)
    print('eta_ini ->', eta_ini)
    print('eta_fin ->', eta_fin)
    print('r_ini  (Mpc/h) ->', r_ini)
    print('r_ini  (Mpc) ->', r_ini * dj.cosmo.h, '\n')


def lightcone_to_healpix(
    lc_data: LightconeData, nside: int, interpolation: bool
) -> jnp.ndarray:
    import jax_healpy
    npix = 12 * nside * nside
    pos = lc_data.pos - lc_data.obs
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    theta = jnp.arctan2(jnp.sqrt(x ** 2 + y ** 2), z)
    phi = jnp.arctan2(y, x)

    weight = (~jnp.isnan(x)).astype(int)

    @jax.sharding.auto_axes(out_sharding=get_sharding_none())
    def hp(theta: jax.Array, phi: jax.Array, weight: jax.Array):
        if not interpolation:
            healpix = jax_healpy.ang2pix(nside, theta, phi)
            hp_dens = jax.ops.segment_sum(weight, healpix, num_segments=npix)
            # this is equivalent to
            # counts = jnp.zeros((npix,))
            # counts = counts.at[healpix].add(jnp.ones_like(theta))
        else:
            pixels, interp_weights = jax_healpy.get_interp_weights(
                nside=nside, theta=theta, phi=phi
            )
            hp_dens = jnp.zeros((npix,))
            hp_dens = hp_dens.at[pixels.T.flatten()].add(
                (interp_weights * weight).T.flatten(),
            )
        return hp_dens

    return hp(theta, phi, weight)
