from discodj.nbody.nbody_dataclass import LightconeData
def simulation_run(setup_data=None):
    """
    Setup for a (potentially multi-GPU) simulation (no gradients).

    Everything is inside a function to not affect the rest of DISCO-DJ on import

    """
    from .config_state import get_global_state_from_args, ConfigState
    from .device_setup import early_setup, device_setup, device_closing

    if setup_data is None:
        state = get_global_state_from_args()
        state = early_setup(state)

    import shutil
    import jax
    import numpy as onp
    from jax_array_info import print_array_stats, sharding_info
    from pathlib import Path

    import dataclasses
    from contextlib import nullcontext
    from matplotlib import pyplot as plt
    from jax.experimental.multihost_utils import sync_global_devices

    if setup_data is None:
        device_name, devices, mode, state = device_setup(state)
    else:
        setup_data: tuple[str, list[jax.Device], str, ConfigState] = setup_data
        device_name, devices, mode, state = setup_data

    from discodj import DiscoDJ
    from discodj.multidevice.utils import get_sharding_none, is_multidevice, is_multiprocess
    from discodj.multidevice.communication import pad_particles, reorder_to_devices, tree_map_by_len
    from discodj.core.array_file_utils import save_all_arrays_to_file, save_data_to_file
    from discodj.core.constants import constants
    from discodj.core.io import save_as_hdf5
    from discodj.core.types import Domain
    from discodj.ics.grf import ICsState
    from discodj.ics.whitenoise import WhiteNoiseFieldState
    from discodj.nbody.nbody_dataclass import Particles, Positions, IntegratorArgs
    from discodj.core.scatter_and_gather import ScatterGatherProperties
    from discodj.gauge_transformations import LinearGRTransferTable, build_linear_gr_state_from_table, \
        get_white_noise_for_gr_state, LinearGRRuntimeState, apply_linear_gr_snapshot
    from discodj.heft.bias_operators import bias_operators, bias_evaluation
    from discodj.heft.bias_data import operators_by_order, HEFTData

    from .load_ics import load_ic_file
    from .cosmology import cosmology_from_config
    from .gauge_transformations import load_gr_transfer_from_file
    from .white_noise_field import get_white_noise_field
    from .run_utils import init_output_dir, global_timing_stats, global_compilation_stats, compile_run_and_profile, \
        JobWatcher
    from .lightcone_utils import lightcone_output_dataclass, lightcone_data_combined, lightcone_to_healpix
    from .plot import plot_slices, plot_powerspectrum
    from .density_field import density_calc, delta_subset, delta_subset_lightcone, save_slice_callback

    watcher = JobWatcher(state.timeout_for_first_sync_point)

    print(devices)
    print(device_name)
    dim = 3
    print(state)

    # Finished with setting up devices and global state
    # Any changes to device setup and global state options should ideally be done before here
    #########################

    output_dir, output_dir_main = init_output_dir(state)

    if state.precision == "double":
        jax.config.update("jax_enable_x64", True)

    cosmo = cosmology_from_config(state)

    dj = DiscoDJ(dim=dim, res=state.res, name=state.name, device=device_name, precision=state.precision, boxsize=state.boxsize,
                 cosmo=cosmo)
    print(dj)

    global_timing_stats.start_timing("calculating timetables and linear ps")

    def dj_setup(dj: DiscoDJ):
        dj = dj.with_timetables()
        if state.linear_ps_file is not None:
            return dj.with_linear_ps(transfer_function="from_file", filename=str(state.linear_ps_file)), dj
        return dj.with_linear_ps(transfer_function="Eisenstein-Hu"), dj

    if state.linear_ps_file and jax.process_index() == 0:
        shutil.copy(state.linear_ps_file, output_dir / "linear_ps_input.txt")

    pk_state, dj = compile_run_and_profile(dj_setup, "dj_setup", dj=dj)
    global_timing_stats.stop_timing("calculating timetables and linear ps")

    print("sync point to make sure we didn't loose any processes up until here")
    sync_global_devices("sync")
    watcher.reached()
    del watcher
    print("passed sync point")

    gr_state: LinearGRRuntimeState | None = None
    if state.gr_correction_lightcone or state.gr_correction_postprocessing:
        assert not state.jax_rng_ics, "GR correction requires ngenic to build correction on low-res whitenoise"
        gr_transfer, gr_transfer_cosmo = load_gr_transfer_from_file(state.gr_correction_file)
        shutil.copy(state.gr_correction_file, output_dir / "gr_correction.asdf")
        if not gr_transfer_cosmo == cosmo:
            print(dataclasses.asdict(cosmo))
            print(dataclasses.asdict(gr_transfer_cosmo))
            raise ValueError("cosmology of gr_correction_file does not match")
        del gr_transfer_cosmo

        gr_whitenoise = get_white_noise_for_gr_state(res=state.gr_correction_res, seed=state.seed_ngenic)

        def build_grstate(gr_transfer: LinearGRTransferTable, white_noise: jax.Array) -> LinearGRRuntimeState:
            return build_linear_gr_state_from_table(table=gr_transfer, white_noise=white_noise,
                                                    res=state.gr_correction_res, boxsize=state.boxsize, cosmo=cosmo)

        gr_state = compile_run_and_profile(build_grstate, "build_grstate", gr_transfer=gr_transfer,
                                           white_noise=gr_whitenoise)

    if not state.ics_file:
        if state.jax_rng_ics:
            def generate_ics_jax_rng(dj: DiscoDJ, pk_state):
                return dj.with_ics(pk_state, seed=state.seed_jaxrng, try_to_jit=False)

            ics: ICsState = compile_run_and_profile(
                generate_ics_jax_rng,
                "generate_ics_jax_rng",
                dj=dj,
                pk_state=pk_state
            )
        else:
            with global_timing_stats.timing("get_white_noise_field (incl. communication setup)"):
                white_noise_field = WhiteNoiseFieldState(
                    field=jax.block_until_ready(get_white_noise_field(dj, state.res, state.seed_ngenic)),
                    domain=Domain.real,
                    res=state.res,
                    dim=dim
                )

            def generate_ics(dj: DiscoDJ, white_noise_field, pk_state):
                return dj.with_ics(pk_state, white_noise=white_noise_field, try_to_jit=False)

            ics: ICsState = compile_run_and_profile(
                generate_ics,
                "generate_ics",
                jit_arguments={"donate_argnames": ("white_noise_field")},
                dj=dj,
                white_noise_field=white_noise_field,
                pk_state=pk_state
            )

            del white_noise_field

        if state.heft:
            bias_parameters = {
                # TODO: replace with properly specified bias parameters
                "b1": 2,
                "b2": 2,
                "s2": 2,
                "d2": 2,
            }

            def calculate_bias_weights(dj: DiscoDJ, ics: ICsState, bias_parameters:dict[str,float]):
                operator_set = operators_by_order[state.heft_bias_order]

                kF = 2 * onp.pi / state.boxsize
                kc = 3
                k_cut = dj.res // kc * kF  # TODO: make a config option
                operators = bias_operators(
                    dj=dj,
                    icstate=ics,
                    a=state.a_end,
                    operators=operator_set,
                    UV_cut=k_cut,
                    grad_order=0,  # TODO
                )
                heft_data = bias_evaluation(
                    bias_parameters=bias_parameters,
                    bias_order=state.heft_bias_order,
                    ops=operators,
                )
                return heft_data

            heft_data: HEFTData = compile_run_and_profile(
                calculate_bias_weights,
                "calculate_bias_weights",
                dj=dj,
                ics=ics,
                bias_parameters=bias_parameters,
            )
        else:
            heft_data = None

        def calculate_lpt(dj: DiscoDJ, ics: ICsState, heft_data: HEFTData | None):
            lpt_state = dj.with_lpt(ics=ics, n_order=state.n_order, convert_to_numpy=False, try_to_jit=False)
            part = dj.with_lpt_ics(lpt_state, state.a_ini)
            part = dataclasses.replace(part, res_pm=state.res_pm)
            if state.heft:
                part = dataclasses.replace(part, heft_data=heft_data)

            npart_per_device = part.npart_total // jax.device_count()
            if state.padded_sim:
                part = part.with_ids()
                part = pad_particles(
                    part, onp.int32(npart_per_device * state.fof_pad_factor)
                )
                part = reorder_to_devices(part, set_worder_alignment=state.worder)
            else:
                part: Particles = dataclasses.replace(
                    part, num=jax.numpy.full((jax.device_count(),), npart_per_device)
                )

            return part


        sim_ini: Particles = compile_run_and_profile(
            calculate_lpt,
            "calculate_lpt",
            jit_arguments={"donate_argnames":["heft_data"]},
            dj=dj,
            ics=ics,
            heft_data=heft_data
        )
        del ics
        del heft_data
    else:
        sim_ini = load_ic_file(Path(state.ics_file), state, dj.cosmo)
        sim_ini = dataclasses.replace(sim_ini, res_pm=state.res_pm)


    sharding_info(sim_ini.vel, "sim_ini.vel")
    sharding_info(sim_ini.psi, "sim_ini.psi")

    method = "pm"
    antialias = 0
    n_resample = 1
    if is_multidevice():
        scatter_gather_props = ScatterGatherProperties(
            res=state.res,
            res_pm=state.res_pm,
            num_devices=state.num_devices,
            use_distributed_scatter_gather=state.use_distributed_scatter_gather,
            do_local_scatter_gather=state.padded_sim,
            use_vjp_gather=state.use_vjp_gather,
            use_vjp_scatter=state.use_vjp_scatter,
            scatter_gather_check=state.scatter_gather_check,
            n_buffer_part_factor=state.n_buffer_part_factor,
            n_buffer_part_fraction=state.n_buffer_part_fraction,
            n_buffer_grid_factor=state.n_buffer_grid_factor,
            n_buffer_grid_fraction=state.n_buffer_grid_fraction,
        )
        if state.padded_sim:
            # should not be needed, but just for safety
            scatter_gather_props.n_buffer_part_factor = 0
            scatter_gather_props.n_buffer_grid_factor = 0
    else:
        scatter_gather_props = None

    if state.save_slice_at_every_timestep:
        def nbody_callback(idx: int, particles: Particles, args: IntegratorArgs):
            save_slice_callback(idx, particles, state, dj, scatter_gather_props, output_dir)
    elif state.snapshot_every_timestep:

        def nbody_callback(idx: int, particles: Particles, args: IntegratorArgs):
            with global_timing_stats.timing("save snapshot during simulation"):
                if state.save_lightcone_data:
                    lc_data, num_particles = lightcone_output_dataclass(
                        sim_out=particles,
                        state=state,
                        dj=dj,
                        compact_array=False,
                    )
                    healpix_out = lightcone_to_healpix(
                        lc_data=lc_data,
                        nside=state.healpix_nside,
                        interpolation=state.healpix_interpolation,
                    )
                    output_data = lc_data
                else:
                    output_data = particles
                output_data = dataclasses.replace(
                    output_data, num=jax.reshard(output_data.num, get_sharding_none())
                )

                def save_callback(callback_data: LightconeData | Particles,idx:int):
                    print(f"nbody callback: step {idx}")
                    if state.save_lightcone_data:
                        file = output_dir_main / f"lightcone_step_{idx}.hdf5"
                    else:
                        file = output_dir_main / f"snapshot_step_{idx}.hdf5"

                    dj.save_as_hdf5(
                        filename=file,
                        data=callback_data,
                        format_str=state.hdf_format,
                        extra_metadata={"simulation_run": state.as_dict()},
                        in_callback=True
                    )
                    print(f"finished writing to {file}")
                    return 1

                jax.experimental.io_callback(
                    save_callback,
                    result_shape_dtypes=jax.numpy.array(1),
                    callback_data=output_data,
                    idx=idx,
                )

    else:
        nbody_callback = None

    def run_nbody(dj: DiscoDJ, sim_ini: Particles, gr_state: LinearGRRuntimeState) -> Particles:
        if state.numsteps:
            out_particles: Particles = dj.run_nbody(
                sim_ini, a_end=state.a_end, n_steps=state.numsteps, res_pm=state.res_pm,
                lightcone=state.lightcone, linear_gr_lightcone=state.gr_correction_lightcone,
                lightcone_crossing_method=state.lightcone_crossing_method,
                linear_gr_state=gr_state, adjoint_method=False, worder=state.worder,
                time_var=state.time_var, stepper=state.stepper, method=method, antialias=antialias,
                grad_kernel_order=state.grad_kernel_order,
                laplace_kernel_order=state.laplace_kernel_order,
                n_resample=n_resample, chunk_size=state.chunk_size,
                scatter_gather_props=scatter_gather_props,
                deconvolve=state.deconvolve_nbody, convert_to_numpy=False, before_ts_callback=nbody_callback
            )
            if out_particles.padded:
                out_particles = reorder_to_devices(out_particles)

        else:
            # skip nbody if numsteps=0
            out_particles = sim_ini


        if state.gr_correction_postprocessing:
            out_particles = compile_run_and_profile(
                apply_linear_gr_snapshot,
                "apply_linear_gr_snapshot",
                particles=out_particles, a=state.a_end, gr_state=gr_state, cosmo=dj.cosmo
            )
        return out_particles

    sim_out: Particles = compile_run_and_profile(
        run_nbody,
        "run_nbody",
        jit_arguments={"donate_argnames": ("sim_ini",)},
        dj=dj,
        sim_ini=sim_ini,
        gr_state=gr_state
    )
    sim_out = dataclasses.replace(
        sim_out,
        mass=constants.particle_mass(dj.cosmo.Omega_m, state.boxsize, sim_out.npart_total)
    )

    if not state.save_hdf5_snapshot and not state.calculate_fof:
        # delete velocity as early as possible in simulations where we don't need it
        sim_out.vel = jax.numpy.zeros(1)

    print_array_stats()

    # jax.profiler.save_device_memory_profile(f"data/memory/memory_{slurm_based_id}.prof")
    global_timing_stats.save_stats(output_dir / "perf.json")

    save_data = {  # smaller data that should always be saved
        "meta": state.as_dict(),
        "cosmo": dataclasses.asdict(dj.cosmo),
        "pk_state": dataclasses.asdict(pk_state),
    }


    if state.calculate_final_density_field:
        density_calc(state, dj, sim_out, scatter_gather_props, save_data, output_dir)

    lightcone_data = None
    if state.lightcone and (state.calculate_fof or state.save_lightcone_data or state.healpix):
        lightcone_data= lightcone_data_combined(
            particles=sim_out,
            state=state,
            dj=dj,
            save_data=save_data,
            compact_array=(state.save_lightcone_data or state.calculate_fof)
        )
        lightcone_data.print_debug()

    if not state.save_final_field and not state.save_hdf5_snapshot:
        # TODO: improve when to remove simulation output
        sim_out.s=None
        sim_out.psi=None
        sim_out.vel=None
        sim_out.id_3d=None

    if state.healpix:
        healpix_out = compile_run_and_profile(
            lightcone_to_healpix,
            "lightcone_to_healpix",
            jit_arguments={"static_argnames": ("nside", "interpolation")},
            lc_data=lightcone_data,
            nside=state.healpix_nside,
            interpolation=state.healpix_interpolation
        )
        if jax.process_index() == 0:
            save_data["healpix"] = {
                "nside": state.healpix_nside,
                "data": healpix_out,
                "num_nan": int(jax.numpy.isnan(healpix_out).sum()),
                "interpolation": state.healpix_interpolation,
            }
            hp_vmax = jax.numpy.nanquantile(healpix_out, 0.95)
            from healpy.newvisufunc import projview
            projview(healpix_out, unit="$z$", min=0, max=hp_vmax)
            fig_hp = plt.gcf()
            fig_hp.savefig(output_dir / "healpix.pdf")


    if state.calculate_fof:
        from .fof_run import calculate_fof, get_fof_config
        from jztree.data import FofCatalogue, expand_particles
        from jztree.config import FofConfig
        from jztree.jax_ext import tree_map_by_len
        from jztree.stats import statistics as fof_stat_manager, Statistics

        fof_dir = output_dir_main / "fof"
        fof_dir.mkdir(exist_ok=True)
        fof_file = fof_dir / f"fof.hdf5"
        a = sim_out.a

        def save_fof_callback(partf: Particles | None, catalogue: FofCatalogue):
            return 0

        with (fof_stat_manager() if state.fof_stats else nullcontext()) as fof_stats:
            fof_cfg = get_fof_config(state)

            if state.lightcone:
                assert lightcone_data is not None
                fof_input = lightcone_data
            else:
                fof_input = dataclasses.replace(
                    sim_out,
                    psi=sim_out.psi.reshape(-1, 3),
                    vel=None,
                    s=None
                )

            partf, catalogue = calculate_fof(state, fof_input, fof_cfg)

            with global_timing_stats.timing("saving fof catalogue"):
                if partf is not None and (
                    state.save_final_field
                    or state.save_hdf5_snapshot
                    or state.save_lightcone_data
                ):
                    if state.lightcone:
                        lightcone_data: LightconeData
                        lightcone_data = partf
                    else:
                        sim_out: Particles
                        sim_out = partf
                print(catalogue)
                print("FoF ngroups (locally):", catalogue.ngroups)
                # catalogue = tree_map_by_len(lambda x: x[:catalogue.ngroups], catalogue, catalogue.count.shape[0])
                dj.save_as_hdf5(fof_file, catalogue, format_str=state.hdf_format)
                exit()
            if state.fof_stats:
                fof_stats: Statistics
                fof_stats.print_suggestions(fof_cfg)
                fof_stats._lock = None
                save_data["fof_stats"] = dataclasses.asdict(fof_stats)

    print_array_stats()

    global_timing_stats.add_general_meta(state.as_dict())

    global_timing_stats.save_stats(output_dir / "perf.json")


    with global_timing_stats.timing("write out save_data"):
        save_all_arrays_to_file(output_dir_main / "arrays/save_data.asdf", save_data, exploded=False)

    if state.save_lightcone_data:
        with global_timing_stats.timing("save hdf5 lightcone"):
            dj.save_as_hdf5(
                filename=output_dir_main / "lightcone.hdf5",
                data=lightcone_data,
                format_str=state.hdf_format,
                extra_metadata={"simulation_run": state.as_dict()},
            )

    if state.save_hdf5_snapshot:
        with global_timing_stats.timing("save hdf5 snapshot"):
            dj.save_as_hdf5(
                filename=output_dir_main / "snapshot.hdf5",
                data=sim_out,
                format_str=state.hdf_format,
                extra_metadata={"simulation_run": state.as_dict()},
            )

    global_timing_stats.save_stats(output_dir / "perf.json")
    if jax.process_index() == 0:
        # only store memory statistics and HLO of rank 0
        global_compilation_stats.save_stats(output_dir)

    device_closing(state, output_dir)
    if not state.uses_job_system and not state.never_plt_show:
        plt.show()

    return output_dir


def cli_entry():
    """
    This is called by the simulation_run cli.
    It returns nothing as the return value is interpreted as the
    return code of the program
    """

    simulation_run()