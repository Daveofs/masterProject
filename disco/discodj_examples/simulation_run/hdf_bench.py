import time

import h5py
import numpy as np
from jax_array_info import sharding_info

from discodj.core.io import save_as_hdf5, _write_hdf5_blocks
from discodj.cosmology.cosmology import Cosmology

a = np.arange(120000000,dtype=np.float32)  # 4.5GB
a=a.reshape((-1,3))
sharding_info(a)

print("start np.save")
start=time.perf_counter()
np.save("/tmp/a.npy",a)
end=time.perf_counter()
print((end-start)*2)

cosmo=Cosmology.from_string_or_dict("Planck15", dtype_num=32, dtype=np.float32)

start=time.perf_counter()
print("start writing")
# save_as_hdf5(
#     x=a,
#     p=a,
#     filename="/tmp/tmp.hdf5",
#     a=1,
#     cosmo=cosmo,
#     boxsize=1
#
# )

with h5py.File("/tmp/a.h5",'w') as f:
    blocks={"Coordinates":a,"Velocities":a}
    _write_hdf5_blocks(blocks,f,format_str="own",cosmo=cosmo,a=1,compressed=True)
end=time.perf_counter()
print(end-start)
