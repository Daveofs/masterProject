from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

def load_clean_params(params_path: Path):
    params = yaml.safe_load(params_path.read_text())
    valid_keys = []
    for k, v in sorted(params.items()):
        try:
            float(v)
            valid_keys.append(k)
        except (ValueError, TypeError):
            continue
    vec = np.array([float(params[k]) for k in valid_keys], dtype=np.float32)
    return vec, valid_keys, params


class ShellAlmDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        lmax: int = 1024,
        max_shells: int = 0,
        verbose: bool = True,
    ):
        data_dir = Path(data_dir)
        self.lmax = lmax

        low_list = []
        high_list = []
        cosmo_list = []
        shell_idx_list = [] 

        subdirs = [d for d in sorted(data_dir.iterdir()) if d.is_dir() and d.name.startswith("cosmo_")]
        if len(subdirs) == 0:
            subdirs = [data_dir]

        leaf_dirs = []
        for sd in subdirs:
            run_dirs = [r for r in sorted(sd.iterdir()) if r.is_dir() and r.name.startswith("run_")]
            leaf_dirs.extend(run_dirs) if run_dirs else leaf_dirs.append(sd)

        total_collected = 0
        param_names = None
        
        # Look for the newly generated precomputed files
        low_file_name = f"low_alms_lmax{lmax}.npz"
        high_file_name = f"high_alms_lmax{lmax}.npz"

        for ld in tqdm(leaf_dirs, desc="Loading precomputed Alm datasets", disable=not verbose):
            if max_shells and total_collected >= max_shells:
                break

            params_yml = ld / "params.yml" if (ld / "params.yml").exists() else ld.parent / "params.yml"
            low_npz = ld / low_file_name
            high_npz = ld / high_file_name

            if not (params_yml.exists() and low_npz.exists() and high_npz.exists()):
                continue

            cosmo_vec, p_names, raw_dict = load_clean_params(params_yml)
            if param_names is None and verbose:
                param_names = p_names
                print(f"\n[YAML Parser] Active Conditioning Vector ({len(p_names)} params)")

            # Instantly load precomputed Alm flat tensors from disk
            low_alms = np.load(low_npz, allow_pickle=False)["alms"]
            high_alms = np.load(high_npz, allow_pickle=False)["alms"]

            n_available = min(low_alms.shape[0], high_alms.shape[0])

            for i in range(n_available):
                if max_shells and total_collected >= max_shells:
                    break

                low_list.append(low_alms[i])
                high_list.append(high_alms[i])
                cosmo_list.append(cosmo_vec)
                shell_idx_list.append(i)

                total_collected += 1

        assert len(low_list) > 0, f"No precomputed files found matching lmax={lmax}. Did you run the preprocessor script?"

        if verbose:
            print(f"[Dataset Loaded] Total items in database: {len(low_list)}")

        self.low_mat = torch.from_numpy(np.stack(low_list))
        self.high_mat = torch.from_numpy(np.stack(high_list))
        self.cosmo_mat = torch.from_numpy(np.stack(cosmo_list))
        self.shell_indices = torch.tensor(shell_idx_list, dtype=torch.float32)

    def __len__(self):
        return self.low_mat.shape[0]

    def __getitem__(self, idx):
        cond = torch.cat([self.cosmo_mat[idx], self.shell_indices[idx:idx+1]])
        return self.low_mat[idx], self.high_mat[idx], cond