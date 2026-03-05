Minimal Flow-Matching training example

Instructions:

1. Ensure the small data folder exists at `/Users/david/testData` and contains:
   - `compressed_shells.npz` (high-res)
   - `shells_nside=512.npz` (low-res)
   - `params.yml` (cosmology key-value pairs)

2. Run a smoke test locally (small nside and few shells):

```bash
conda activate flow_env                                    
python ml/train_flow_matching.py
```

Notes:
- This is a proof-of-concept trainer that regresses `u_t = x1 - x0` with a small MLP.
- For realistic runs use larger `nside_small` or patch-based UNets and more training data.