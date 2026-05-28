# Usage

## General Notes

`simulation_run` and the code in this directory provides an interface to run forward DISCO-DJ simulation without gradient evaluation.

**NOTE**: The default option so far were mostly for development and are not always ideal for "actual" simulations. We will improve this in the next steps.

For now one call to `simulation_run` runs exactly one simulation. But when running many simulations that only differ in e.g. rng seed, it would be much faster to run them in the same process (as compilation often takes as long as the simulation). So this will be added at a later time.

## Configuration

All parameters can be set using commandline arguments or a config file (or both):

```text
usage: simulation_run [-h] [--name NAME] [--run-mode RUN_MODE] [--boxsize BOXSIZE] [--res RES]
                      [--res-pm RES_PM] [--n-order N_ORDER] [--a-ini A_INI] [--a-end A_END]
                      [--cosmo COSMO] [--precision PRECISION] [--numsteps NUMSTEPS]
                      [--num-chunks NUM_CHUNKS] [--stepper STEPPER] [--time-var TIME_VAR]
                      [--grad-kernel-order GRAD_KERNEL_ORDER]
                      [--laplace-kernel-order LAPLACE_KERNEL_ORDER] [--worder WORDER]
                      [--single-device] [--no-single-device] [--linear-ps-file LINEAR_PS_FILE]
                      [--ics-file ICS_FILE] [--skip-ics-header-check] [--no-skip-ics-header-check]
                      [--padded-sim] [--no-padded-sim] [--gr-correction-res GR_CORRECTION_RES]
                      [--gr-correction-file GR_CORRECTION_FILE] [--gr-correction-postprocessing]
                      [--no-gr-correction-postprocessing] [--gr-correction-lightcone]
                      [--no-gr-correction-lightcone] [--double-precision-timetables]
                      [--no-double-precision-timetables]
                      [--n-buffer-part-factor N_BUFFER_PART_FACTOR]
                      [--n-buffer-part-fraction N_BUFFER_PART_FRACTION]
                      [--n-buffer-grid-factor N_BUFFER_GRID_FACTOR]
                      [--n-buffer-grid-fraction N_BUFFER_GRID_FRACTION] [--lightcone]
                      [--no-lightcone] [--lightcone-size-factor LIGHTCONE_SIZE_FACTOR]
                      [--lightcone-size-fraction LIGHTCONE_SIZE_FRACTION]
                      [--lightcone-crossing-method LIGHTCONE_CROSSING_METHOD] [--rsd] [--no-rsd]
                      [--healpix] [--no-healpix] [--healpix-nside HEALPIX_NSIDE]
                      [--use-distributed-scatter-gather] [--no-use-distributed-scatter-gather]
                      [--use-vjp-scatter] [--no-use-vjp-scatter] [--use-vjp-gather]
                      [--no-use-vjp-gather] [--scatter-gather-check] [--no-scatter-gather-check]
                      [--deconvolve-nbody] [--no-deconvolve-nbody] [--deconvolve-density-field]
                      [--no-deconvolve-density-field] [--slice-axis SLICE_AXIS] [--fNL-test]
                      [--no-fNL-test] [--use-fof-callback] [--no-use-fof-callback]
                      [--calculate-final-density-field] [--no-calculate-final-density-field]
                      [--save-slice-at-every-timestep] [--no-save-slice-at-every-timestep]
                      [--calculate-fof] [--no-calculate-fof] [--fof-link-length FOF_LINK_LENGTH]
                      [--fof-pad-factor FOF_PAD_FACTOR] [--fof-alloc-fac-nodes FOF_ALLOC_FAC_NODES]
                      [--fof-alloc-fac-ilist FOF_ALLOC_FAC_ILIST]
                      [--fof-alloc-fac-distr-links FOF_ALLOC_FAC_DISTR_LINKS]
                      [--fof-npart-min FOF_NPART_MIN] [--fof-stats] [--no-fof-stats]
                      [--timeout-for-first-sync-point TIMEOUT_FOR_FIRST_SYNC_POINT] [--dump-xla]
                      [--no-dump-xla] [--dump-xla-in-all-processes] [--no-dump-xla-in-all-processes]
                      [--save-final-field] [--no-save-final-field] [--save-hdf5-snapshot]
                      [--no-save-hdf5-snapshot] [--quiet-compile] [--no-quiet-compile]
                      [--never-plt-show] [--no-never-plt-show] [--powerspectrum] [--no-powerspectrum]
                      [--seed-ngenic SEED_NGENIC] [--seed-jaxrng SEED_JAXRNG] [--jax-rng-ics]
                      [--no-jax-rng-ics] [--locked] [--no-locked] [--double] [--z-ini Z_INI]
                      [--export-config]
                      [config]
```

Current default options for parameter can be found in [`config_state.py`](./config_state.py), but they will change until the public release of "new" DISCO-DJ. You can find the exact parameters used in `your_simulation/meta.json`.

```bash
simulation_run --run-mode gpu --lightcone --res 4096 --res-pm 4096 --boxsize 8000 --name testrun_4096_lc
```
Alternatively as a config file:

```toml
run_mode = "gpu"
boxsize = 8000.0
res = 4096
res_pm = 4096
lightcone = true
```

```bash
simulation_run config.toml
```


## Cosmology

For now I did not yet write an interface to provide a custom set of cosmological parameters to the simulation. For now the easiest way to run simulations with custom cosmological parameters is editing [`discodj/cosmology/predefined_cosmologies.py`](../../discodj/cosmology/predefined_cosmologies.py), adding an entry for the new set and specifying this new name with `--cosmology`.

## Config files

You can use `--export-config` as a starting point:

```
$ simulation_run --res 128 --res-pm 128 --boxsize 2000 --numsteps 32 --cosmo Planck18EEBAOSN --run-mode gpu --no-dump-xla --export-config

config.toml:
run_mode = "gpu"
boxsize = 2000.0
res = 128
res_pm = 128
cosmo = "Planck18EEBAOSN"
numsteps = 32
dump_xla = false
```


## Output

`simulation_run` assumes that there is a `data` folder in the current directory that is on a filesystem intended for storing large data.

`data/output` will have one folder per simulation that contains all output (based on `--name` and job ID).

`data/dump*` contains the full XLA dump, which is very useful to debug memory usage and other details, but can create a huge amount of small files (which many filesystems can't handle well). This can be disabled with `--no-dump-xla`


## Jobscripts

See [the jobscripts/ folder](../../../jobscripts/) for examples on what I am using.

All jobscripts are written in a way that they eventually call `simulation_run $@`. Therefore one can provide all options to sbatch like `sbatch jobscript.sh --run-mode gpu --name something`

### Nvidia Profiler

TODO

## Environment variables, XLA flags and other global options

JAX/XLA/CUDA/NCCL have an endless amount of vaguely documented flags and environment variables.

I set some of them in the jobscript and more in [device_setup.py](./device_setup.py). 
The benefit/effect of most of them is not entirely clear and they should later be compared against https://guides.lw1.at/all-xla-options/, https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/docs/GPU_performance.md, the JAX documentation and benchmarked. 

Some that are for sure useful:
- `NCCL_DEBUG=TRACE` this enables verbose logging whenever NCCL initializes the distributed communication or sets up another distributed communication operation. This output shows information that e.g. confirms if Infiniband is used for communication between nodes. 
