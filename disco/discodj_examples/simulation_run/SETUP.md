# Setup

## GSL

Either way we need GSL, so load a fitting module. The exact version doesn't really matter (`module load GSL` on MUSICA, `module load gsl` on Leonardo).

This module also needs to be loaded in the jobscript, the same way it would be in other C/C++ code using GSL.

## DJ CUDA Kernels

To avoid complications, the easiest is to disable them completely for now (this will be cleaned up later):

```diff
diff --git a/CMakeLists.txt b/CMakeLists.txt
index 6c92fd0..f0cd48c 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -48,7 +48,7 @@ set_target_properties(_discodj_native PROPERTIES
 install(TARGETS _discodj_native LIBRARY DESTINATION "discodj_native")
 
 # Optional CUDA-based scatter kernel for JAX FFI.
-option(DISCO_DJ_ENABLE_CUDA "Build CUDA scatter kernel" ON)
+option(DISCO_DJ_ENABLE_CUDA "Build CUDA scatter kernel" OFF)
 option(DISCO_DJ_CUDA_INPLACE "Place CUDA scatter library in source tree for editable installs" ON)
 option(DISCO_DJ_REQUIRE_CUDA "Fail configuration if CUDA scatter kernel is not built" OFF)
```


## Create a Python venv with cuda-enabled JAX

We need at least Python 3.10 and want to always use a separate virtual environment for DISCO-DJ: 
```bash
# something like
module load Python
python -m venv ~/my_venvs/discodj/
source ~/my_venvs/discodj/bin/activate
```

### If CUDA 13 is supported by cluster

e.g. on [MUSICA](https://docs.asc.ac.at/systems/musica.html)

In this case things are really easy as we don't need any CUDA libraries provided by the system and can use the nvidia pip packages exclusively.

So a `pip install -U "jax[cuda13]"` as the [JAX documentation recommends](https://docs.jax.dev/en/latest/installation.html) works directly as expected.

And we can install JAX, DISCO-DJ and other used packages in one step:

```bash
# inside your fresh venv
pip install -r pip/requirements-gpu-cuda13.txt
# or 
uv sync --extra gpu_cuda13 -vv
```

Afterwards you only need to install jz-tree ([see below](#)), but also for this we don't need further dependencies.

### If you need to use CUDA 12

Unfortunately the same isn't possible as the [nvidia-cuda-nvcc-cu12](https://pypi.org/project/nvidia-cuda-nvcc-cu12/) pip package somehow doesn't include nvcc.

#### If you don't need FoF/jz-tree/cuda kernels

If we don't need to compile custom CUDA kernels (e.g. for jz-tree/FoF), we can ignore this issue and just use the CUDA 12 version of JAX (`pip install --upgrade "jax[cuda12]"`).

To then install JAX, DISCO-DJ and other required packages, you can use:

```bash
pip install -r pip/requirements-gpu-cuda12.txt
# or
uv sync --extra gpu_cuda12 -vv
```

#### If you do need FoF on CUDA 12

Then things get a bit complicated as we need to compile CUDA kernels, but can't use the pip-provided nvcc. Therefore we need to use an up-to-date CUDA 12 version from \*somewhere\*. In theory the CUDA modules provided by the cluster (and a `pip install --upgrade "jax[cuda12-local]"` afterwards) should be a solution, but at least on Leonardo they are far too outdated ([these](https://docs.jax.dev/en/latest/installation.html#pip-installation-nvidia-gpu-cuda-installed-locally-harder) are the minimum requirements for JAX).

So one solution we found (which is unfortunately very inconvenient) is to use the CUDA libraries from conda-forge. So instead of creating a "normal" python venv, do the following:

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-Linux-x86_64.sh
# this will install miniforge to ~/miniforge3
# When asked "Proceed with initialization?" select "no", so that miniforge doesn't modify the .bashrc
eval "$(~/miniforge3/bin/conda 'shell.bash' 'hook')"
conda create --name discodj
conda activate discodj
conda install pip
# now we have a conda-venv and have to install CUDA here
# This combination seems to be working:
export CONDA_OVERRIDE_CUDA="12.9"
conda install -c conda-forge cuda-nvcc cuda-version=12 cudnn nccl libcufft cuda-cupti libcublas libcusparse
pip install nanobind cmake # these might not be needed at this point, but it can't hurt
pip install --upgrade "jax[cuda12-local]"
```

Now at this time we should have a working JAX setup that uses the CUDA libraries from conda-forge. We can test this by submitting a jobscript that does something like this:
```bash
nvidia-smi

eval "$(~/miniforge3/bin/conda 'shell.bash' 'hook')"
conda activate discodj

python -c "import jax; print(jax.devices())"
```

If we don't get warnings about missing CUDA libraries or not found GPUs, but instead an array like `[CudaDevice(id=0), CudaDevice(id=1), CudaDevice(id=2), CudaDevice(id=3)]` we are on a good path and can continue with DISCO-DJ as expected. We can then install DISCO-DJ and the other normal packages like this:

```bash
pip install -r pip/requirements-gpu-cuda12_local.txt
```

## Install jz-tree

Let's assume we have cloned the source code into `~/jz-tree/`

```bash
git clone git@github.com:jstuecker/jz-tree.git ~/jz-tree/
```

These steps should ideally be done on the same GPU nodes where the code is run later (so that nvcc can use the native GPU arch). Check [jz-tree documentation](https://jstuecker.github.io/jztree/installation.html) on how to specify the GPU arch manually.

If we did the CUDA-13 version of steps above, this installation is also trivial.

A 
```bash
pip install -e . --no-build-isolation
```
is all we need. We could also add `--extra cuda_tree` to the `uv sync` above to let uv do this in one go.

If this doesn't work, check the jz-tree README and double-check that `pip install scikit-build-core setuptools_scm nanobind jax[cuda13]` is installed.

If you are using the local CUDA 12 version (either from your cluster modules or from conda-forge) the same should hopefully work:
```bash
# make sure these are installed 
pip install scikit-build-core setuptools_scm nanobind cmake
pip install -e . --no-build-isolation 
# if this doesn't work, try again with `export CUDA_LOCAL=1`
```

If this works, feel free to move on to [USAGE.md](./USAGE.md) for the next steps.

## CPU version

For fast local testing, one can also run low-resolution simulations on CPU. 
Don't trust that the JAX compiler will have the same memory usage and behavior on CPU as on distributed GPUs, but to check for obvious bugs and trying out options, this is still useful. Also FoF/jz-tree uses CUDA kernels and doesn't work at all on CPU.

You can install using
```bash
pip install -r pip/requirements-local-testing.txt
```
for a minimal CPU version or 
```bash
uv sync --extra pyqt --extra personal --extra pyvista --extra cuda_13_only --extra cuda_tree --extra docs --extra discoeb --extra healpix
```
to get the identical venv that I use for development locally.

### CPU Run-Modes

```bash
simulation_run --run-mode singlecpu
```

Runs a "normal" single device simulation

```bash
simulation_run --run-mode cpu
# or
simulation_run
```

Runs using 16 "fake" CPU devices. This is very useful for fast local development, but has notable differences to "real" multi-GPU simulations. E.g. this attaches all devices to one process, which is not what we do on the cluster.

```bash
mpirun -np 4 simulation_run
```

This also uses CPU devices for a distributed simulation, but unlike before they are actually separate Python processes per CPU using the "distributed"
 version of JAX. This is the closest we can get to a distributed simulation without actually using multiple GPUs, but still only a quick test and not always trustworthy.

## Output data structure

For now the output paths are hardcoded so that everything gets written to `./data/` (relative to the pwd where `simulation_run` gets run). 
As this output can get very large, `data/` should point to a large storage space (e.g. by creating a symlink there `ln -s $DATA/somewhere ./data`)

## Using Slurm Jobscripts

Inside the `jobscripts/` directory of the main directory you can find the jobscripts I am using to run DISCO-DJ on [MUSICA](https://docs.asc.ac.at/systems/musica.html) and [LEONARDO](https://docs.hpc.cineca.it/hpc/leonardo.html#leonardo-card). They should be easy to adapt to other Slurm-based HPC centers.

The slurm scripts assume that a `jobscripts/output/` directory exists where all slurm output logs will be written into (split by rank).