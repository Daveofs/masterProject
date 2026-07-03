"""DeepSphere map-correction: DISCO-DJ low-res shells -> CosmoGrid high-res shells.

This module uses the *exact* graph-CNN from deepsphere-cosmo-tf1
(``deepsphere.models.deepsphere`` / ``cgcnn``) — no reimplementation — configured
as a **fully-convolutional, constant-resolution** network so it maps a full
HEALPix shell to a corrected full HEALPix shell (map -> map regression).

How the correction task maps onto deepsphere(cgcnn)
---------------------------------------------------
``cgcnn`` is normally a classifier/regressor (map -> a few scalars): it pools the
sphere down and applies a global ``statistics`` layer. We disable all of that:

    * p          = [1, 1, ...]   -> no pooling (constant nside through the net)
    * statistics = None          -> no global pooling of spatial info
    * M          = []            -> no fully-connected head
    * F[-1]      = 1             -> single output channel (one value per pixel)
    * loss       = 'l2'          -> per-pixel regression

With this config ``cgcnn._inference`` returns a (batch, Npix) tensor — the
corrected map — and the l2 loss compares it against the high-res target map.

The Chebyshev graph convolutions, weight init, Laplacian rescaling, batch-norm,
training loop, checkpointing, etc. are all the original DeepSphere code.

Note: deepsphere is TensorFlow (TF1 API via tensorflow.compat.v1). Importing this
module's model functions pulls in TensorFlow. The data loaders are pure numpy.

Usage
-----
    from MLP import load_shell_pairs, build_model, train

    X, Y, norm = load_shell_pairs("/path/to/grid", nside=512)
    model = build_model(nside=512, n_layers=5)
    train(model, X, Y, val_frac=0.1)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

# DeepSphere uses the TF1 API (tf.compat.v1 + tf.layers.batch_normalization),
# which was removed under Keras 3 (TF >= 2.16). On such TF we route tf.keras to
# the tf_keras (Keras 2) shim. But only do this if tf_keras is actually installed
# — e.g. the NGC TF 2.15 container has native Keras 2 and NO tf_keras package, and
# forcing the flag there breaks Keras import. Must be set before TF is imported.
import importlib.util as _ilu
if _ilu.find_spec("tf_keras") is not None:
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import healpy as hp


# ---------------------------------------------------------------------------
# DeepSphere import plumbing
# ---------------------------------------------------------------------------

# Location of the deepsphere-cosmo-tf1 checkout. Override with $DEEPSPHERE_PATH.
DEEPSPHERE_PATH = os.environ.get(
    "DEEPSPHERE_PATH", "/users/damrein/deepsphere-cosmo-tf1"
)


def _ensure_deepsphere_on_path():
    if DEEPSPHERE_PATH not in sys.path:
        sys.path.insert(0, DEEPSPHERE_PATH)


def _load_deepsphere():
    """Import the deepsphere TF model (this pulls in TensorFlow)."""
    _ensure_deepsphere_on_path()
    try:
        from deepsphere import models
        from deepsphere.data import LabeledDataset
    except ImportError as e:
        raise ImportError(
            f"Could not import deepsphere from {DEEPSPHERE_PATH!r}. "
            "Set $DEEPSPHERE_PATH to your deepsphere-cosmo-tf1 checkout."
        ) from e
    return models, LabeledDataset


# ---------------------------------------------------------------------------
# Data loading: pair low-res (DISCO) and high-res (CosmoGrid) shells
# ---------------------------------------------------------------------------

def _read_shells(npz_path: Path, key: str = "shells") -> np.ndarray:
    """Load a (n_shells, Npix) array of HEALPix maps from an .npz file."""
    data = np.load(str(npz_path))
    arr = data[key]
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# Partial-sphere patching (for high nside via DeepSphere's `order` mechanism)
# ---------------------------------------------------------------------------
#
# A full-sphere graph at nside=2048 has 50M nodes -> a single Chebyshev conv is
# infeasible. DeepSphere's solution (utils.nside2indexes) is to split the sphere
# into ``12*order**2`` equal patches of ``(nside/order)**2`` pixels. In NESTED
# ordering the children of each low-res (nside=order) superpixel are contiguous,
# so the split is a pure reshape, and — by HEALPix symmetry — every patch has the
# same internal graph, so ONE Laplacian (built on the first patch) is shared by
# all. Each patch becomes an independent training sample.
#
# Trade-off: graph edges crossing patch boundaries are dropped, so the receptive
# field is truncated at patch borders (negligible for order small relative to
# nside, i.e. large patches). This is the original DeepSphere behaviour.

def n_patches(order: int) -> int:
    return 12 * order * order


def resolve_F_hidden(F_hidden, n_layers: int) -> list:
    """Resolve the hidden feature widths, matching get_correction_params' default."""
    if F_hidden is None:
        F_hidden = [16, 32, 64, 32][: max(n_layers - 1, 1)]
        while len(F_hidden) < n_layers - 1:
            F_hidden.append(F_hidden[-1])
    return list(F_hidden)


def safe_gpu_batch(nside: int, order: int, F_hidden, margin: float = 0.9) -> int:
    """Largest batch size that keeps TF's GPU sparse matmul under its 2^31 limit.

    TF's SparseTensorDenseMatMul GPU kernel requires output.shape[1]*nnz(L) < 2^31.
    In chebyshev5 that is nnz(L) * (max_Fin * batch). The HEALPix graph has ~9
    nonzeros per node, so nnz(L) ~= 9 * patch_npix. Returns the max batch (>=1)
    satisfying max_Fin * batch * nnz(L) < margin * 2^31.
    """
    patch_npix = hp.nside2npix(nside) // n_patches(order)
    nnz = 9 * patch_npix
    max_F = max(list(F_hidden) + [1]) if F_hidden else 1
    limit = int(margin * (2 ** 31))
    return max(limit // (max_F * nnz), 1)


def _gpu_batch_for(nside, order, F_hidden, n_layers, margin=0.9):
    """safe_gpu_batch with F_hidden resolved to the model's actual widths."""
    return safe_gpu_batch(nside, order, resolve_F_hidden(F_hidden, n_layers), margin)


def map_to_patches(maps: np.ndarray, order: int) -> np.ndarray:
    """(N, Npix) NESTED maps -> (N * 12*order^2, patch_npix) patches."""
    if order <= 1:
        return maps
    N, npix = maps.shape
    npatch = n_patches(order)
    patch_npix = npix // npatch
    return maps.reshape(N, npatch, patch_npix).reshape(N * npatch, patch_npix)


def patches_to_maps(patches: np.ndarray, order: int, n_maps: int) -> np.ndarray:
    """(n_maps * 12*order^2, patch_npix) patches -> (n_maps, Npix) NESTED maps."""
    if order <= 1:
        return patches
    npatch = n_patches(order)
    patch_npix = patches.shape[1]
    return patches.reshape(n_maps, npatch, patch_npix).reshape(n_maps, npatch * patch_npix)


# ---------------------------------------------------------------------------
# Per-shell overdensity normalization
# ---------------------------------------------------------------------------
#
# HEALPix density shells span ~5 orders of magnitude in amplitude across redshift
# (near shells are dense, far shells nearly empty). A single global (mean, std)
# is dominated by the dense shells and makes faint shells unrecoverable. Instead
# we normalize each shell to the dimensionless overdensity
#
#     delta = rho / mean(rho) - 1        (per shell, mass-conserving)
#
# which is comparable across all shells/redshifts. Both low and high are expressed
# relative to the LOW shell mean, so a corrected map inverts with that same mean:
#
#     rho_corrected = mean_low * (1 + delta_pred)
#
# A single global delta_scale (std of delta over a sample) rescales inputs to ~unit
# range for the network. delta itself is scale-free, so one global scale is fine.

def _shell_means(maps: np.ndarray) -> np.ndarray:
    """Per-shell means with zeros guarded to 1 (keepdims for broadcasting)."""
    m = maps.mean(axis=-1, keepdims=True)
    return np.where(m == 0, 1.0, m)


def overdensity_forward(low: np.ndarray, high: np.ndarray, delta_scale: float,
                        residual: bool):
    """Physical (low, high) shells -> normalized (X, Y) in overdensity space.

    Both are taken relative to the per-shell LOW mean. Y is the correction
    (delta_high - delta_low) if residual else delta_high, divided by delta_scale.
    """
    m = _shell_means(low)
    dlow = low / m - 1.0
    dhigh = high / m - 1.0
    X = (dlow / delta_scale).astype(np.float32)
    Y = ((dhigh - dlow) if residual else dhigh) / delta_scale
    return X, Y.astype(np.float32)


def estimate_delta_scale(low_maps: np.ndarray) -> float:
    """Global std of the per-shell overdensity, used to rescale to ~unit range."""
    d = low_maps / _shell_means(low_maps) - 1.0
    return float(d.std() + 1e-12)


def load_shell_pairs(
    data_dir,
    nside: int,
    low_name: str = "shells_nside=512.npz",
    high_name: str = "compressed_shells.npz",
    nest: bool = True,
    order: int = 1,
    max_pairs: Optional[int] = None,
    standardize: bool = True,
    residual: bool = False,
    verbose: bool = True,
):
    """Load paired (low, high) HEALPix shells across cosmologies.

    Walks ``data_dir`` for cosmology subdirectories (``cosmo_*`` or ``run_*``),
    each holding a low-res shell stack (DISCO-DJ) and a high-res stack
    (CosmoGrid). Returns flat arrays of per-shell maps ready for LabeledDataset.

    Parameters
    ----------
    data_dir : path
        Root directory containing cosmology subdirectories.
    nside : int
        Target HEALPix nside. Maps at a different nside are ud_grade'd to it.
    low_name, high_name : str
        Filenames of the low- and high-res shell ``.npz`` stacks within each run.
    nest : bool
        HEALPix ordering of the stored maps (must match the Laplacian; deepsphere
        builds its graph in NESTED ordering by default).
    order : int
        Partial-sphere split factor. order=1 keeps full-sphere maps; order>1
        splits each map into 12*order**2 patches of (nside/order)**2 pixels so
        high nside (e.g. 2048) is tractable. Patches become independent samples.
    max_pairs : int, optional
        Cap on the number of (low, high) shell pairs (useful for quick tests).
        Applied to whole maps BEFORE patch splitting.
    standardize : bool
        Standardize low maps to zero-mean/unit-std using global statistics.
    residual : bool
        If True, the target Y is the (standardized) high-minus-low residual, so
        the model learns a correction added on top of the low map. Otherwise Y is
        the (standardized) high map directly.

    Returns
    -------
    X : (N, Npix) float32 — low-res input maps (standardized if requested)
    Y : (N, Npix) float32 — targets (high map or residual, standardized)
    norm : dict — normalization statistics needed to invert at inference time
    """
    data_dir = Path(data_dir)
    npix = hp.nside2npix(nside)

    subdirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and (d.name.startswith("cosmo_") or d.name.startswith("run_"))
    )
    if not subdirs:
        subdirs = [data_dir]

    low_maps: list[np.ndarray] = []
    high_maps: list[np.ndarray] = []

    def _to_nside(m: np.ndarray) -> np.ndarray:
        m = m.astype(np.float32)
        if m.shape[0] != npix:
            order_in = "NESTED" if nest else "RING"
            m = hp.ud_grade(m, nside, order_in=order_in, order_out=order_in)
        return m.astype(np.float32)

    n_pairs = 0
    for sd in subdirs:
        # Allow either flat (run dir) or nested (cosmo/run) layouts.
        run_dirs = [r for r in sorted(sd.iterdir())
                    if r.is_dir() and r.name.startswith("run_")] if sd.is_dir() else []
        leaf_dirs = run_dirs if run_dirs else [sd]
        for ld in leaf_dirs:
            low_npz, high_npz = ld / low_name, ld / high_name
            if not (low_npz.exists() and high_npz.exists()):
                continue
            low_stack = _read_shells(low_npz)
            high_stack = _read_shells(high_npz)
            n_shells = min(low_stack.shape[0], high_stack.shape[0])
            for i in range(n_shells):
                low_maps.append(_to_nside(low_stack[i]))
                high_maps.append(_to_nside(high_stack[i]))
                n_pairs += 1
                if max_pairs is not None and n_pairs >= max_pairs:
                    break
            if max_pairs is not None and n_pairs >= max_pairs:
                break
        if max_pairs is not None and n_pairs >= max_pairs:
            break

    if n_pairs == 0:
        raise RuntimeError(
            f"No (low, high) shell pairs found under {data_dir} "
            f"(looking for {low_name!r} + {high_name!r})."
        )

    X = np.stack(low_maps).astype(np.float32)   # (N, Npix)
    H = np.stack(high_maps).astype(np.float32)  # (N, Npix)
    if verbose:
        print(f"[load_shell_pairs] {n_pairs} shell pairs | nside={nside} | "
              f"Npix={npix:,} | X={X.shape} H={H.shape}")

    # Per-shell overdensity normalization (scale-free across redshift).
    delta_scale = estimate_delta_scale(X)
    norm: dict = {"nside": nside, "nest": nest, "residual": residual,
                  "mode": "overdensity", "delta_scale": delta_scale, "order": order,
                  "npix": npix, "patch_npix": npix // n_patches(order)}
    X, Y = overdensity_forward(X, H, delta_scale, residual)
    if verbose:
        print(f"[load_shell_pairs] overdensity norm | delta_scale={delta_scale:.4g} | "
              f"residual={residual}")

    # Split full-sphere maps into patches so high nside is tractable (order>1).
    if order > 1:
        X = map_to_patches(X, order)
        Y = map_to_patches(Y, order)
        if verbose:
            print(f"[load_shell_pairs] order={order}: split into "
                  f"{X.shape[0]:,} patches of {X.shape[1]:,} pixels each")

    return X, Y, norm


def invert_prediction(pred: np.ndarray, x_low: np.ndarray, norm: dict) -> np.ndarray:
    """Map a model prediction back to a physical corrected map using ``norm``.

    x_low is the PHYSICAL low map(s) (per-shell mean is recovered from it). pred is
    the network output in scaled-overdensity space. Inversion is mass-conserving:
    rho = mean_low * (1 + delta_pred), with delta_pred = delta_low + pred*scale for
    residual mode, else pred*scale.
    """
    scale = norm["delta_scale"]
    m = _shell_means(x_low)                 # per-shell low mean, (…,1)
    dpred = pred * scale
    if norm.get("residual", False):
        dpred = (x_low / m - 1.0) + dpred   # add back the low overdensity
    return m * (1.0 + dpred)


# ---------------------------------------------------------------------------
# Model: the exact deepsphere(cgcnn), configured for map -> map regression
# ---------------------------------------------------------------------------

def get_correction_params(
    nside: int,
    order: int = 1,
    n_layers: int = 5,
    F_hidden: Optional[list] = None,
    K: int = 5,
    batch_norm: bool = True,
    num_epochs: int = 40,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
    total_steps: Optional[int] = None,
    lr_final_frac: float = 0.05,
    regularization: float = 0.0,
    eval_frequency: int = 50,
    distributed: bool = False,
    dir_name: str = "deepsphere_correction",
    verbose: bool = True,
) -> dict:
    """Build the kwargs dict for ``models.deepsphere`` in map->map mode.

    Fully-convolutional, constant-resolution config: no pooling, no statistics
    layer, no fully-connected head, single output channel, l2 loss. The result is
    a per-pixel regression model (corrected map = f(low map)).
    """
    import tensorflow.compat.v1 as tf  # same API deepsphere itself uses

    # Optimizer factory. For multi-GPU we wrap Adam in Horovod's DistributedOptimizer
    # (deepsphere's training() calls optimizer.compute_gradients -> Horovod inserts
    # the gradient allreduce) and linearly scale the LR by the number of workers.
    if distributed:
        import horovod.tensorflow as hvd
        _size = hvd.size()

        def _make_optimizer(lr):
            base = tf.train.AdamOptimizer(lr * _size, beta1=0.9, beta2=0.999, epsilon=1e-8)
            return hvd.DistributedOptimizer(base)
    else:
        def _make_optimizer(lr):
            return tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)

    F_hidden = resolve_F_hidden(F_hidden, n_layers)
    assert len(F_hidden) == n_layers - 1, \
        f"F_hidden must have n_layers-1={n_layers-1} entries, got {len(F_hidden)}"

    F = list(F_hidden) + [1]            # last layer outputs 1 channel per pixel
    K_list = [K] * n_layers
    bn = [batch_norm] * n_layers
    # No pooling => every level is the same nside, so build_laplacians derives
    # pooling factors p=[1, 1, ...] on its own. deepsphere() computes (L, p) from
    # nsides internally, so p must NOT be passed here. We need
    # len(nsides) == n_conv_layers + 1.
    nsides = [nside] * (n_layers + 1)

    # Partial-sphere: build the graph on a single patch of (nside/order)^2 pixels.
    # Every patch shares this Laplacian (HEALPix symmetry). order=1 -> full sphere.
    indexes = None
    if order > 1:
        _ensure_deepsphere_on_path()
        from deepsphere import utils as ds_utils
        indexes = ds_utils.nside2indexes(nsides, order)

    params = dict(
        nsides=nsides,
        indexes=indexes,               # None=full sphere; else one patch's nodes
        F=F,
        K=K_list,
        batch_norm=bn,
        M=[],                          # no fully-connected head
        conv="chebyshev5",
        pool="average",               # unused (p=1) but must be valid
        activation="relu",
        statistics=None,               # keep spatial resolution -> map output
        loss="l2",                     # per-pixel regression
        regularization=regularization,
        dropout=1,
        num_epochs=num_epochs,
        batch_size=batch_size,
        eval_frequency=eval_frequency,
        # Smooth exponential decay across the WHOLE run: lr -> lr*lr_final_frac at
        # the final step. (The original deepsphere used decay_steps=1/rate=0.999,
        # which collapses the LR to ~0 within a few thousand steps and stalls
        # training — wrong for the many-step regime here.)
        scheduler=lambda step: tf.train.exponential_decay(
            learning_rate, step,
            decay_steps=max(total_steps or 1, 1), decay_rate=lr_final_frac),
        optimizer=_make_optimizer,
        dir_name=dir_name,
    )
    if verbose:
        patch_npix = hp.nside2npix(nside) // n_patches(order)
        scope = "full sphere" if order == 1 else \
            f"order={order} -> {n_patches(order)} patches of {patch_npix:,} px"
        lr_end = learning_rate * lr_final_frac
        print(f"[get_correction_params] nside={nside} | {scope} | layers={n_layers} | "
              f"F={F} | K={K} | loss=l2 | lr {learning_rate:.1e}->{lr_end:.1e} "
              f"over {total_steps or '?'} steps")
    return params


def build_model(nside: int, **kwargs):
    """Instantiate the exact deepsphere(cgcnn) model for map->map correction."""
    models, _ = _load_deepsphere()
    params = get_correction_params(nside, **kwargs)
    return models.deepsphere(**params)


# ---------------------------------------------------------------------------
# Training (uses deepsphere's own fit() loop)
# ---------------------------------------------------------------------------

def train(model, X: np.ndarray, Y: np.ndarray, val_frac: float = 0.1,
          shuffle: bool = True, seed: int = 0):
    """Train via deepsphere's native ``fit``.

    Parameters
    ----------
    model : deepsphere(cgcnn) instance from ``build_model``.
    X : (N, Npix) low-res input maps.
    Y : (N, Npix) target maps (high or residual).
    val_frac : fraction of samples held out for validation.
    """
    _, LabeledDataset = _load_deepsphere()

    N = X.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N) if shuffle else np.arange(N)
    n_val = max(int(round(val_frac * N)), 1) if val_frac > 0 else 0
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    training = LabeledDataset(X[train_idx], Y[train_idx])
    validation = LabeledDataset(X[val_idx], Y[val_idx]) if n_val > 0 else None

    print(f"[train] train={len(train_idx)} val={len(val_idx)} samples")
    # deepsphere.fit returns:
    #   accuracies_validation, losses_validation, losses_training, t_step
    acc_val, loss_val, loss_train, t_step = model.fit(training, validation)
    return acc_val, loss_val, loss_train, t_step


@np.errstate(all="ignore")
def predict_maps(model, X: np.ndarray, x_low_phys: np.ndarray, norm: dict) -> np.ndarray:
    """Run the model on standardized inputs X and return physical corrected maps."""
    pred = model.predict(X)
    pred = np.asarray(pred)
    return invert_prediction(pred, x_low_phys, norm)


# ===========================================================================
# Streaming dataset: scale to 20k-170k shells without preloading everything
# ===========================================================================
#
# deepsphere's in-memory LabeledDataset holds every map in RAM, which is
# impossible at high shell counts (one nside=2048 shell = 201 MB). This streaming
# dataset reads ONE .npz file pair at a time (each file = all shells of a run,
# e.g. 69 shells), splits it into patches, yields patch batches, then moves on.
# Peak RAM is bounded by a single file pair (~30 GB at nside=2048) regardless of
# how many files (= how many shells) the full dataset spans.
#
# It is duck-typed to what cgcnn.fit() needs: ``.N``, ``.shuffled``,
# ``.iter(batch_size)`` (an endless generator of equal-size (X, Y) batches) and
# ``.get_all_data()`` (only call this on a SMALL validation set — it loads all).

def _peek_shells_shape(npz_path, key: str = "shells"):
    """Read a (n_shells, Npix) array's shape from an .npz WITHOUT loading it."""
    import zipfile
    import numpy.lib.format as fmt
    with zipfile.ZipFile(str(npz_path)) as z:
        name = key if key.endswith(".npy") else key + ".npy"
        with z.open(name) as f:
            version = fmt.read_magic(f)
            shape, _fortran, _dtype = fmt._read_array_header(f, version)
    return shape


def build_file_pairs(data_root, test_cosmo: Optional[str] = None,
                     low_name: str = "shells_nside=2048.npz",
                     high_name: str = "compressed_shells.npz",
                     include_only_test: bool = False) -> list:
    """Enumerate (low_npz, high_npz) file pairs across the cosmology grid.

    Each pair corresponds to one run directory (= all its shells). For
    leave-one-out training, pass test_cosmo to exclude it (include_only_test=False)
    or to select ONLY it (include_only_test=True, e.g. to build a validation set).
    """
    data_root = Path(data_root)
    cosmos = sorted(d for d in data_root.iterdir()
                    if d.is_dir() and d.name.startswith("cosmo_"))
    if not cosmos:
        cosmos = [data_root]

    pairs = []
    for c in cosmos:
        is_test = (test_cosmo is not None and c.name == test_cosmo)
        if include_only_test and not is_test:
            continue
        if (not include_only_test) and is_test:
            continue
        run_dirs = [r for r in sorted(c.iterdir())
                    if r.is_dir() and r.name.startswith("run_")] if c.is_dir() else []
        for ld in (run_dirs if run_dirs else [c]):
            low, high = ld / low_name, ld / high_name
            if low.exists() and high.exists():
                pairs.append((low, high))
    return pairs


class StreamingShellDataset:
    """Disk-streaming, patch-yielding dataset for deepsphere(cgcnn).fit().

    Streams one .npz file pair at a time (each = all shells of a run), so peak RAM
    is one file pair regardless of total dataset size. Standardization uses global
    stats estimated from a small sample of files (or supplied via ``norm``).
    """

    def __init__(self, file_pairs, nside, order: int = 1, nest: bool = True,
                 residual: bool = False, norm: Optional[dict] = None,
                 shuffle: bool = True, seed: int = 0,
                 stat_sample_files: int = 2, max_eval_patches: int = 0,
                 verbose: bool = True):
        self.file_pairs = list(file_pairs)
        if not self.file_pairs:
            raise RuntimeError("StreamingShellDataset: no file pairs given.")
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.order = order
        self.nest = nest
        self.residual = residual
        self._shuffle = shuffle
        self._rng = np.random.RandomState(seed)
        # Cap on patches returned by get_all_data (validation); 0 = no cap.
        self.max_eval_patches = max_eval_patches
        self.verbose = verbose

        # Total #patches across all files (peek headers — cheap, no data loaded).
        self._shells_per_file = []
        for low, _high in self.file_pairs:
            n = _peek_shells_shape(low)[0]
            self._shells_per_file.append(int(n))
        total_shells = int(sum(self._shells_per_file))
        self._N = total_shells * n_patches(order)

        # Global normalization stats.
        if norm is not None:
            self.norm = norm
        else:
            self.norm = self._estimate_norm(stat_sample_files)
        self.norm.setdefault("order", order)
        self.norm.setdefault("nside", nside)
        self.norm.setdefault("residual", residual)
        self.norm.setdefault("mode", "overdensity")
        self.norm.setdefault("nest", nest)

        if verbose:
            print(f"[StreamingShellDataset] {len(self.file_pairs)} files | "
                  f"{total_shells} shells | order={order} -> {self._N:,} patches | "
                  f"delta_scale={self.norm.get('delta_scale'):.4g} residual={residual}")

    # --- interface expected by cgcnn.fit ---
    @property
    def N(self):
        return self._N

    @property
    def shuffled(self):
        return self._shuffle

    def get_all_data(self):
        """Load ALL file pairs into memory as patches. Use only on small splits.

        If ``max_eval_patches`` is set, a random subset of that many patches is
        returned — validation over the full split would otherwise be very costly
        (it runs at every eval_frequency during fit()).
        """
        Xs, Ys = [], []
        for fi in range(len(self.file_pairs)):
            Xm, Ym = self._load_file_pair(fi)
            Xs.append(map_to_patches(Xm, self.order))
            Ys.append(map_to_patches(Ym, self.order))
        X, Y = np.concatenate(Xs), np.concatenate(Ys)
        if self.max_eval_patches and X.shape[0] > self.max_eval_patches:
            sel = np.random.RandomState(0).choice(
                X.shape[0], self.max_eval_patches, replace=False)
            X, Y = X[sel], Y[sel]
        return X, Y

    def iter(self, batch_size=1):
        return self.__iter__(batch_size)

    def __iter__(self, batch_size=1):
        order_files = np.arange(len(self.file_pairs))
        while True:  # endless: fit() decides how many steps to pull
            if self._shuffle:
                self._rng.shuffle(order_files)
            for fi in order_files:
                Xm, Ym = self._load_file_pair(int(fi))
                Xp = map_to_patches(Xm, self.order)
                Yp = map_to_patches(Ym, self.order)
                idx = np.arange(Xp.shape[0])
                if self._shuffle:
                    self._rng.shuffle(idx)
                for b in range(0, len(idx) - batch_size + 1, batch_size):
                    bi = idx[b:b + batch_size]
                    yield Xp[bi], Yp[bi]

    # --- internals ---
    def _estimate_norm(self, n_files: int) -> dict:
        # Estimate the single global delta_scale from a few files' low shells.
        sample = self.file_pairs[: max(n_files, 1)]
        lows = [self._read_maps(low) for low, _high in sample]
        X = np.concatenate(lows)
        return {"nside": self.nside, "nest": self.nest, "residual": self.residual,
                "mode": "overdensity", "order": self.order, "npix": self.npix,
                "patch_npix": self.npix // n_patches(self.order),
                "delta_scale": estimate_delta_scale(X)}

    def _read_maps(self, npz_path) -> np.ndarray:
        """Load a (n_shells, npix) stack, resampling to target nside if needed."""
        maps = np.asarray(np.load(str(npz_path))["shells"], dtype=np.float32)
        if maps.shape[1] != self.npix:
            order_str = "NESTED" if self.nest else "RING"
            maps = np.stack([
                hp.ud_grade(m, self.nside, order_in=order_str, order_out=order_str)
                for m in maps
            ]).astype(np.float32)
        return maps

    def _load_file_pair(self, fi: int):
        """Return per-shell overdensity-normalized (X_maps, Y_maps) for one file."""
        low, high = self.file_pairs[fi]
        X = self._read_maps(low)
        H = self._read_maps(high)
        return overdensity_forward(X, H, self.norm["delta_scale"], self.residual)


def train_streaming(model, train_ds: "StreamingShellDataset",
                    val_ds: "StreamingShellDataset"):
    """Train deepsphere(cgcnn) on streaming datasets (train + small in-memory val).

    The validation set is loaded fully (get_all_data) by fit(), so keep it small
    (a couple of files). Training streams file-by-file and never preloads.
    """
    print(f"[train_streaming] train patches={train_ds.N:,} | "
          f"val patches={val_ds.N:,}")
    return model.fit(train_ds, val_ds)


def train_horovod(model, train_ds: "StreamingShellDataset", num_epochs: int,
                  batch_size: int, hvd, log_every: int = 50):
    """Data-parallel training of deepsphere(cgcnn) with Horovod.

    Bypasses deepsphere.fit() to run a custom synchronous SGD loop: pin each rank
    to its local GPU, broadcast rank-0's initial weights, then stream this rank's
    data shard and run the (Horovod-wrapped) op_train, which all-reduces gradients
    every step. Only rank 0 logs and writes the checkpoint that predict() restores.

    The model must have been built with ``distributed=True`` so its optimizer is a
    hvd.DistributedOptimizer. ``train_ds`` should already be this rank's shard.
    """
    import os
    import tensorflow.compat.v1 as tf

    rank, size = hvd.rank(), hvd.size()

    # Broadcast op must live in the model's graph (references its variables).
    # deepsphere.build_graph() calls graph.finalize(); temporarily un-finalize so
    # we can append the broadcast op (finalize is only a guard against accidental
    # graph growth — safe to lift here).
    model.graph._unsafe_unfinalize()
    with model.graph.as_default():
        bcast = hvd.broadcast_global_variables(0)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    num_steps = max(int(num_epochs * train_ds.N / batch_size), 1)
    if rank == 0:
        print(f"[train_horovod] {size} workers | {num_steps:,} steps/worker | "
              f"batch/worker={batch_size} (effective {batch_size * size}) | "
              f"shard patches={train_ds.N:,}", flush=True)

    losses = []
    with tf.Session(graph=model.graph, config=config) as sess:
        sess.run(model.op_init)
        sess.run(bcast)                     # sync all workers to rank-0 weights
        train_iter = train_ds.iter(batch_size)
        t0 = time.time()
        for step in range(1, num_steps + 1):
            xb, yb = next(train_iter)
            _, loss = sess.run(
                [model.op_train, model.op_loss],
                feed_dict={model.ph_data: xb, model.ph_labels: yb,
                           model.ph_training: True})
            if rank == 0 and step % log_every == 0:
                rate = step / (time.time() - t0)
                print(f"  step {step:,}/{num_steps:,} | loss={loss:.4e} | "
                      f"{rate:.1f} steps/s", flush=True)
                losses.append(float(loss))
        # Rank 0 saves a checkpoint so the apply stage's predict() can restore it.
        if rank == 0:
            ckpt_dir = model._get_path('checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            model.op_saver.save(sess, os.path.join(ckpt_dir, 'model'),
                                global_step=num_steps)
            print(f"[train_horovod] saved checkpoint to {ckpt_dir}", flush=True)
    return losses
