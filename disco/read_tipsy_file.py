import numpy as np
import jax

def read_tipsy(nBody_file_in, Lbox):
    try:
        f = open(nBody_file_in, 'r')
    except IOError:
        print('IOERROR: N-body tipsy file does not exist!')
        print('Define par.files.partfile_in = "/path/to/file"')
        exit()

    #header
    print("Reading tipsy-file...")
    p_header_dt = np.dtype([('a','>d'),('npart','>i'),('ndim','>i'),('ng','>i'),('nd','>i'),('ns','>i'),('buffer','>i')])
    p_header = np.fromfile(f, dtype=p_header_dt, count=1, sep='')
    print("File contains %d particles." % p_header['npart'])

    #particles
    p_dt = np.dtype([('mass','>f'),("x",'>f'),("y",'>f'),("z",'>f'),("vx",'>f'),("vy",'>f'),("vz",'>f'),("eps",'>f'),("phi",'>f')])
    p = np.fromfile(f, dtype=p_dt, count=int(p_header['npart']), sep='')

    #from tipsy units to [0,Lbox] in units of Lbox
    p['x']=Lbox*(p['x']+0.5)
    p['y']=Lbox*(p['y']+0.5)
    p['z']=Lbox*(p['z']+0.5)

    if jax.process_index() == 0:
        print('Reading tipsy-file done!')
    return p, p_header


def read_tipsy_binary(nbody_file_in, Lbox, header_only=False):
    """Binary-mode tipsy reader (fixes the 'r' vs 'rb' issue in read_tipsy).

    Parameters
    ----------
    nbody_file_in : path to the PKDGRAV tipsy file
    Lbox          : box size (same units as the output positions)
    header_only   : if True, read only the header and return (None, header)

    Returns
    -------
    (p, p_header) where p is the structured particle array with x/y/z in
    [0, Lbox] and vx/vy/vz in raw PKD units, or (None, p_header) when
    header_only=True.
    """
    p_header_dt = np.dtype([
        ('a', '>d'), ('npart', '>i'), ('ndim', '>i'),
        ('ng', '>i'), ('nd', '>i'), ('ns', '>i'), ('buffer', '>i'),
    ])
    p_dt = np.dtype([
        ('mass', '>f'), ('x', '>f'), ('y', '>f'), ('z', '>f'),
        ('vx', '>f'), ('vy', '>f'), ('vz', '>f'), ('eps', '>f'), ('phi', '>f'),
    ])
    with open(nbody_file_in, 'rb') as f:
        p_header = np.frombuffer(f.read(p_header_dt.itemsize), dtype=p_header_dt)
        if jax.process_index() == 0:
            print(f"[read_tipsy_binary] {nbody_file_in}: {int(p_header['npart'][0])} particles")
        if header_only:
            return None, p_header
        npart = int(p_header['npart'][0])
        p = np.frombuffer(f.read(p_dt.itemsize * npart), dtype=p_dt).copy()

    # Convert positions from [-0.5, 0.5] (tipsy normalised) to [0, Lbox]
    p['x'] = Lbox * (p['x'] + 0.5)
    p['y'] = Lbox * (p['y'] + 0.5)
    p['z'] = Lbox * (p['z'] + 0.5)
    if jax.process_index() == 0:
        print('[read_tipsy_binary] done')
    return p, p_header


def read_tipsy_chunks(nbody_file_in, Lbox, chunk_size=10**6):
    """Stream a PKDGRAV tipsy binary file in chunks, yielding structured arrays.

    Each yielded chunk has x/y/z in [0, Lbox].  Useful for very large files
    (e.g. N=2048^3) where reading everything into RAM at once is infeasible.
    """
    p_header_dt = np.dtype([
        ('a', '>d'), ('npart', '>i'), ('ndim', '>i'),
        ('ng', '>i'), ('nd', '>i'), ('ns', '>i'), ('buffer', '>i'),
    ])
    p_dt = np.dtype([
        ('mass', '>f'), ('x', '>f'), ('y', '>f'), ('z', '>f'),
        ('vx', '>f'), ('vy', '>f'), ('vz', '>f'), ('eps', '>f'), ('phi', '>f'),
    ])
    with open(nbody_file_in, 'rb') as f:
        p_header = np.frombuffer(f.read(p_header_dt.itemsize), dtype=p_header_dt)
        num_particles = int(p_header['npart'][0])
        if jax.process_index() == 0:
            print(f"[read_tipsy_chunks] streaming {num_particles} particles "
                  f"in chunks of {chunk_size}")
        offset = 0
        while offset < num_particles:
            count = min(chunk_size, num_particles - offset)
            raw = np.frombuffer(f.read(p_dt.itemsize * count), dtype=p_dt).copy()
            raw['x'] = Lbox * (raw['x'] + 0.5)
            raw['y'] = Lbox * (raw['y'] + 0.5)
            raw['z'] = Lbox * (raw['z'] + 0.5)
            yield raw
            offset += count


def tipsy_to_hdf5(
    tipsy_file: str,
    output_hdf5: str,
    Lbox: float,
    a: float,
    h: float | None = None,
    omega_m: float | None = None,
    omega_b: float | None = None,
    omega_lambda: float | None = None,
    redshift: float | None = None,
) -> str:
    """Convert a PKDGRAV tipsy binary file to a PKDGRAV-compatible HDF5 file.

    The output HDF5 is structured so that ``load_ic_file`` (from
    ``discodj_examples.simulation_run.load_ics``) picks up the pkdgrav branch
    automatically via the ``"PKDGRAV version"`` header attribute.

    Coordinates are stored in the [-0.5, 0.5] pkdgrav convention so that
    ``load_ic_file`` applies the canonical ``(coords + 0.5) * boxsize`` transform.

    Parameters
    ----------
    tipsy_file    : path to the input PKDGRAV tipsy binary file
    output_hdf5   : path for the output .hdf5 file
    Lbox          : box size (length units consistent with the simulation)
    a             : scale factor at IC time
    h, omega_m, omega_b, omega_lambda : cosmological params written to the HDF5
        Cosmology group for ``load_ic_file`` header checks.  Pass None to omit
        the Cosmology group and let ``load_ic_file`` skip header checks.
    redshift      : if None, derived as 1/a - 1

    Returns
    -------
    output_hdf5 path (str), ready to pass to ``load_ic_file``.
    """
    import h5py

    if redshift is None:
        redshift = 1.0 / a - 1.0

    p, p_header = read_tipsy_binary(tipsy_file, Lbox)
    num_particles = int(p_header['npart'][0])

    # Convert [0, Lbox] → [-0.5, 0.5] so load_ic_file gets the expected range
    coords = np.stack([p['x'], p['y'], p['z']], axis=1).astype(np.float32)
    coords = coords / Lbox - 0.5
    velocities = np.stack([p['vx'], p['vy'], p['vz']], axis=1).astype(np.float32)
    del p

    # Infer Lagrangian grid IDs by rounding each particle to its nearest grid
    # point (valid at high redshift where displacements << 1 grid cell).
    # Sequential IDs (np.arange) would be wrong because tipsy particle order
    # is NOT row-major Lagrangian order — load_ic_file uses id_3d to compute q,
    # so wrong IDs → wrong q → wrong displacement psi = pos - q.
    res = int(round(num_particles ** (1.0 / 3)))
    coords01 = coords + 0.5  # → [0, 1]
    ix = np.round(coords01[:, 0] * res).astype(np.int64) % res
    iy = np.round(coords01[:, 1] * res).astype(np.int64) % res
    iz = np.round(coords01[:, 2] * res).astype(np.int64) % res
    particle_ids = (ix * res * res + iy * res + iz).astype(np.int64)

    with h5py.File(output_hdf5, 'w') as f:
        hdr = f.create_group("Header")
        hdr.attrs["BoxSize"]       = Lbox
        hdr.attrs["HubbleParam"]   = float(h) if h is not None else 0.0
        hdr.attrs["Redshift"]      = np.array([redshift])
        hdr.attrs["Time"]          = np.array([a])
        hdr.attrs["NumPart_Total"] = np.array([0, num_particles, 0, 0, 0, 0],
                                               dtype=np.int64)
        hdr.attrs["PKDGRAV version"] = "tipsy_converted"  # triggers pkdgrav branch

        if all(v is not None for v in (h, omega_m, omega_b, omega_lambda)):
            cosmo_grp = f.create_group("Cosmology")
            cosmo_grp.attrs["Omega_b"]      = float(omega_b)
            cosmo_grp.attrs["Omega_m"]      = float(omega_m)
            cosmo_grp.attrs["Omega_lambda"] = float(omega_lambda)

        pg = f.create_group("PartType1")
        pg.create_dataset("Coordinates", data=coords)
        pg.create_dataset("Velocities",  data=velocities)
        pg.create_dataset("ParticleIDs", data=particle_ids)

    if jax.process_index() == 0:
        print(f"[tipsy_to_hdf5] '{tipsy_file}' -> '{output_hdf5}' "
              f"({num_particles} particles)")
    return output_hdf5


def tipsy_to_hdf5_chunked(
    tipsy_file: str,
    output_hdf5: str,
    Lbox: float,
    a: float,
    h: float | None = None,
    omega_m: float | None = None,
    omega_b: float | None = None,
    omega_lambda: float | None = None,
    redshift: float | None = None,
    chunk_size: int = 10**6,
) -> str:
    """Chunked version of ``tipsy_to_hdf5`` for very large tipsy files.

    Streams the tipsy file in ``chunk_size``-particle blocks and writes
    directly into pre-allocated HDF5 datasets, avoiding the need to hold the
    entire particle array in RAM.  See ``tipsy_to_hdf5`` for full parameter
    documentation.
    """
    import h5py

    if redshift is None:
        redshift = 1.0 / a - 1.0

    _, p_header = read_tipsy_binary(tipsy_file, Lbox, header_only=True)
    num_particles = int(p_header['npart'][0])

    with h5py.File(output_hdf5, 'w') as f:
        hdr = f.create_group("Header")
        hdr.attrs["BoxSize"]       = Lbox
        hdr.attrs["HubbleParam"]   = float(h) if h is not None else 0.0
        hdr.attrs["Redshift"]      = np.array([redshift])
        hdr.attrs["Time"]          = np.array([a])
        hdr.attrs["NumPart_Total"] = np.array([0, num_particles, 0, 0, 0, 0],
                                               dtype=np.int64)
        hdr.attrs["PKDGRAV version"] = "tipsy_converted"

        if all(v is not None for v in (h, omega_m, omega_b, omega_lambda)):
            cosmo_grp = f.create_group("Cosmology")
            cosmo_grp.attrs["Omega_b"]      = float(omega_b)
            cosmo_grp.attrs["Omega_m"]      = float(omega_m)
            cosmo_grp.attrs["Omega_lambda"] = float(omega_lambda)

        pg = f.create_group("PartType1")
        coords_ds = pg.create_dataset("Coordinates",
                                      shape=(num_particles, 3), dtype=np.float32)
        vels_ds   = pg.create_dataset("Velocities",
                                      shape=(num_particles, 3), dtype=np.float32)
        ids_ds    = pg.create_dataset("ParticleIDs",
                                      shape=(num_particles,),   dtype=np.int64)

        res = int(round(num_particles ** (1.0 / 3)))
        offset = 0
        for chunk in read_tipsy_chunks(tipsy_file, Lbox, chunk_size):
            n = len(chunk['x'])
            coords = np.stack([chunk['x'], chunk['y'], chunk['z']], axis=1).astype(np.float32)
            coords = coords / Lbox - 0.5  # → [-0.5, 0.5] for load_ic_file
            vels   = np.stack([chunk['vx'], chunk['vy'], chunk['vz']], axis=1).astype(np.float32)
            # Infer Lagrangian grid IDs by rounding to nearest grid point.
            # Sequential IDs would be wrong: tipsy order is not row-major
            # Lagrangian order, and load_ic_file uses id_3d to compute q.
            coords01 = coords + 0.5  # → [0, 1]
            ix = np.round(coords01[:, 0] * res).astype(np.int64) % res
            iy = np.round(coords01[:, 1] * res).astype(np.int64) % res
            iz = np.round(coords01[:, 2] * res).astype(np.int64) % res
            chunk_ids = (ix * res * res + iy * res + iz).astype(np.int64)
            coords_ds[offset:offset + n] = coords
            vels_ds[offset:offset + n]   = vels
            ids_ds[offset:offset + n]    = chunk_ids
            offset += n

    if jax.process_index() == 0:
        print(f"[tipsy_to_hdf5_chunked] '{tipsy_file}' -> '{output_hdf5}' "
              f"({num_particles} particles)")
    return output_hdf5