# AI-Enhanced Fast Cosmological Simulations for Accurate Large-Scale Structure Inference

Master's thesis, ETH Zurich — David Amrein
Advisors: Jozef Bucko, Alexandre Refregier

Map-level, simulation-based inference needs far more forward-modelled lightcones than
full $N$-body suites can supply. This repository builds and evaluates a cheaper route:
run a fast particle-mesh solver ([Disco-DJ](https://github.com/cosmo-sims/DISCO-DJ)) on
initial conditions matched to the reference suite
([CosmoGridV1](https://cosmogrid.ai/), run with PkdGrav3), then *learn* the small-scale
content the mesh could not resolve.

Formally, we seek a correction `f` with

```
f( Map_disco )  ≈  Map_CosmoGridV1
```

where both maps come from the same cosmology and the same realisation of the initial
density field, so `f` has only the force-solver difference to model.

## What is here

| Directory | Contents |
|---|---|
| `pkdgrav/` | Initial-condition generation. Rebuilds each CosmoGridV1 cosmology's linear input (CDM+baryon transfer function, `σ8_cb`, `Ω_r`) from the run's own stored CONCEPT perturbations, patches `cosmology.par`, and drives PkdGrav3 to produce one backscaled particle load. Also the shell collector and the shell-builder validation driver. |
| `shell_builder/` | The lightcone shell builder added to Disco-DJ's time integration. Reproduces PkdGrav3's moving-lightcone crossing condition on CosmoGridV1's exact 69-shell radial grid, with periodic box replication, in constant memory (two particle snapshots + S count maps). |
| `disco/` | Disco-DJ run drivers: tipsy→HDF5 IC conversion and the SLURM array that produced the 198-cosmology production set. |
| `ml/transfer/` | Correction pipeline 1 — deterministic per-multipole transfer-function rescaling in harmonic space, with an MLP emulator of `T(ℓ, s)` conditioned on cosmology. |
| `ml/unet/` | Correction pipeline 2 — conditional flow-matching U-Net transporting each map patch from low to high fidelity. |
| `ml/diffusion/` | Correction pipeline 3 — conditional EDM diffusion model generating the missing small-scale residual from noise. |
| `ml/analysis/` | The shared, pipeline-neutral evaluation library: full-sky `C_ℓ`, patch tiling and reconstruction, one-point moments, weak-lensing convergence (UFalcon). |
| `ml/preprocess/` | Dataset construction: paired patch extraction, `a_ℓm` precomputation. |
| `vis/` | Every figure in the thesis. Each script is standalone and writes one plot. |
| `report/` | The thesis source (`report.tex`) and its figures. |

`ml/almflow/` and `ml/sphereflow/` are earlier model variants, kept for the record;
they are not part of the reported results.

## Design rules

Two conventions are worth knowing before reading the code.

**The pipeline modules are decoupled on purpose.** `ml/transfer/`, `ml/unet/` and
`ml/diffusion/` never import from one another. Small shared helpers are duplicated
rather than factored out, so any one pipeline can be changed without silently moving
the others. The single sanctioned shared module is `ml/analysis/`, which is what makes
the three pipelines comparable figure-by-figure.

**Every driver is paired with a SLURM script.** The Python entry point runs unchanged
from a laptop-scale `16³` test to a production allocation; only the surrounding
`run_*.sh` changes.

## Reproducing the results

The stages run in data order. Each `run_*.sh` carries its own resource header.

```bash
# 1. initial conditions: linear input -> backscaled particle load
pkdgrav/run_gen_all_transfer_functions.sh     # CDM+baryon T(k) per cosmology
pkdgrav/run_IC_preparation.sh                 # patch cosmology.par, derive sigma8_cb, Omega_r
pkdgrav/run_pkdgrav_gen_all_cscs.sh           # PkdGrav3 IC generation -> CosmoML.tipsy

# 2. fast lightcones on CosmoGridV1's shell grid
disco/run_disco_gen_all_cscs.sh               # SLURM array, 198 cosmologies

# 3. corrections
ml/transfer/run_transfer.sh
ml/unet/run_flow.sh
ml/diffusion/run_diffusion.sh
```

Evaluation is shared: all three pipelines are scored on the same 30 held-out
cosmologies, at a common `N_side = 512`, `ℓ_max = 1500` footing.

## Data

The CosmoGridV1 reference suite (~110 TB) is public and is not redistributed here.
Simulation outputs, shell maps and trained checkpoints live on scratch, not in the
repository; only code and the thesis are version-controlled.

## Requirements

Python 3.11+, JAX (GPU), PyTorch, `healpy`, `numpy`, `matplotlib`, `classy`, and a
PkdGrav3 build. The two solvers use separate virtual environments — Disco-DJ needs a
matched JAX/CUDA stack, PkdGrav3 its own compiled toolchain. Production runs target
NVIDIA GH200 nodes on CSCS Alps.
