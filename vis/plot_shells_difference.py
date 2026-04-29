from pathlib import Path
import healpy as hp
import numpy as np
from visualize import plot_shells


if __name__ == "__main__":
    p1 = Path("/Users/david/projects/outputs/plots/shell_builder_validation/built_maps/CosmoML-built-shell_step=00104_z-high=0.7010866_z-low=0.6713487_chatgbt.fits")
    p2 = Path("/Users/david/projects/outputs/plots/shell_builder_validation/built_maps/CosmoML-built-shell_step=00104_z-high=0.7010866_z-low=0.6713487_old.fits")
    out_dir = Path("/Users/david/projects/outputs/plots/shells")
    out_dir.mkdir(parents=True, exist_ok=True)
    diff_name = "diff.fits"
    out_path = out_dir / diff_name
    
    m1 = hp.read_map(p1, nest=False, dtype=np.float32)
    m2 = hp.read_map(p2, nest=False, dtype=np.float32)
    if m1.size != m2.size:
        raise SystemExit(f"Input FITS have different sizes: {m1.size} vs {m2.size}")

    m = m2 + m1
    
    if out_path.exists():
        out_path.unlink()
    hp.write_map(str(out_path), m, nest=False)
    print(f"Wrote difference FITS to {out_path}")

    # Plot the difference map (do not re-normalize the difference)
    plot_shells(
        npz_path=out_path,
        z_bin=int(10),
        nside=int(128),
        output_dir=out_dir,
        plot_logarithmic=True,
        name=f"diff"
    )
else:
    raise SystemExit("No input files provided")

