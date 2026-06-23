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

        # 1. Store open disk-pointers, NOT data
        self.low_mmaps = []
        self.high_mmaps = []
        
        # 2. Address book: global_idx -> (file_pointer_idx, row_inside_that_file)
        self.sample_addresses = [] 
        
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
        file_pointer_idx = 0
        
        low_file_name = f"low_alms_lmax{lmax}.npy"
        high_file_name = f"high_alms_lmax{lmax}.npy"

        for ld in tqdm(leaf_dirs, desc="Indexing .npy Memory Maps", disable=not verbose):
            if max_shells and total_collected >= max_shells:
                break

            params_yml = ld / "params.yml" if (ld / "params.yml").exists() else ld.parent / "params.yml"
            low_npy = ld / low_file_name
            high_npy = ld / high_file_name

            if not (params_yml.exists() and low_npy.exists() and high_npy.exists()):
                continue

            cosmo_vec, p_names, _ = load_clean_params(params_yml)
            if param_names is None and verbose:
                param_names = p_names
                print(f"\n[YAML Parser] Active Conditioning Vector ({len(p_names)} params)")

            # Instantly mount the file to virtual memory (Costs ~100 bytes of RAM per file)
            low_mmap = np.load(low_npy, mmap_mode='r')
            high_mmap = np.load(high_npy, mmap_mode='r')

            self.low_mmaps.append(low_mmap)
            self.high_mmaps.append(high_mmap)

            n_available = min(low_mmap.shape[0], high_mmap.shape[0])

            for i in range(n_available):
                if max_shells and total_collected >= max_shells:
                    break

                self.sample_addresses.append((file_pointer_idx, i))
                cosmo_list.append(cosmo_vec)
                shell_idx_list.append(i)

                total_collected += 1

            file_pointer_idx += 1

        assert len(self.sample_addresses) > 0, f"No .npy files found for lmax={lmax}."
        
        if verbose:
            print(f"[Dataset Mounted] Total items mapped on disk: {len(self.sample_addresses)}")

        # Only the tiny conditioning floats get loaded into actual RAM
        self.cosmo_mat = torch.from_numpy(np.stack(cosmo_list))
        self.shell_indices = torch.tensor(shell_idx_list, dtype=torch.float32)

    def __len__(self):
        return len(self.sample_addresses)

    def __getitem__(self, idx):
        file_idx, row_idx = self.sample_addresses[idx]

        # Slicing the mmap triggers an OS kernel interrupt to fetch strictly this 1 row
        # Wrapping in np.array() forces a memory copy so PyTorch doesn't complain about read-only buffers.
        x0 = torch.from_numpy(np.array(self.low_mmaps[file_idx][row_idx]))
        x1 = torch.from_numpy(np.array(self.high_mmaps[file_idx][row_idx]))

        cond = torch.cat([self.cosmo_mat[idx], self.shell_indices[idx:idx+1]])
        return x0, x1, cond