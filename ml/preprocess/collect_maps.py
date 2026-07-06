# ---------------------------------------------------------------------------
# Data: mmap per-shell streaming of (tcorr, high) pairs
# ---------------------------------------------------------------------------

def build_runs(data_root, test_cosmo, nside, include_test, prefix="low"):
    """(input_npy, high_npy, cosmo_vec) per run that has the prepared dataset.

    prefix='low'   -> raw DISCO input   (single-model 'direct' formulation)
    prefix='tcorr' -> T-corrected input ('residual' formulation)
    """
    from matplotlib.path import Path
    from pathlib import Path
    import numpy as np
    data_root = Path(data_root)
    runs = []
    for c in sorted(d for d in data_root.iterdir()
                    if d.is_dir() and d.name.startswith("cosmo_")):
        if (not include_test) and c.name == test_cosmo:
            continue
        for ld in sorted(r for r in c.iterdir()
                         if r.is_dir() and r.name.startswith("run_")) or [c]:
            tc = ld / f"{prefix}_shells_nside={nside}.npy"
            hi = ld / f"high_shells_nside={nside}.npy"
            if not (tc.exists() and hi.exists()):
                continue
            pf = ld / "params.yml"
            vec = _cosmo_vector(pf) if pf.exists() else np.zeros(1, np.float32)
            runs.append((tc, hi, vec))
    return runs


def _cosmo_vector(params_yml):
    import yaml
    from matplotlib.path import Path
    import numpy as np
    p = yaml.safe_load(Path(params_yml).read_text())
    keys = sorted(k for k, v in p.items() if _is_num(v))
    return np.array([float(p[k]) for k in keys], dtype=np.float32)


def _is_num(v):
    try:
        float(v); return True
    except (ValueError, TypeError):
        return False
