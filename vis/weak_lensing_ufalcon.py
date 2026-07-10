from UFalcon import construct_maps, utils
import numpy as np
import healpy as hp
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
import astropy


if __name__ == "__main__":

    lightcone_path = "/capstor/scratch/cscs/damrein/cosmogridv1/cosmo_000122/run_0/compressed_shells.npz"
    n_z = "/capstor/scratch/cscs/damrein/redshift_distribution/desy3_nz_metacal_bin1.txt"

    print(f"Loading lightcone data from: {lightcone_path}")
    # Load the compressed shell data
    shell_info = np.load(lightcone_path, mmap_mode="r")['shell_info']
    lower_z = shell_info['lower_z']
    upper_z = shell_info['upper_z']
    print(f"Lower redshift: {lower_z}")
    print(f"Higher redshift: {upper_z}")
    shells = np.load(lightcone_path,mmap_mode="r")['shells'] 

    print(f"Loading redshift distribution from: {n_z}")
    # Load the redshift distribution
    n_z = np.loadtxt(n_z)
    plt.plot(n_z[:, 0], n_z[:, 1])
    plt.xlabel("Redshift")
    plt.ylabel("n(z)")
    plt.title("Redshift Distribution")
    plt.savefig("/capstor/scratch/cscs/damrein/outputs/plots/weak_lensing/redshift_distribution.png", dpi=300)

    #Set up cosmology of sims
    eV = astropy.units.eV
    Omega_m= 0.26
    Omega_b = 0.0493
    sigma_8 = 0.84
    Neff = 3.046
    H0 = 67.36
    O_nu =0.001422554 #neutrino energy density
    cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m-O_nu, Neff=Neff,Ob0=Omega_b, m_nu=0.02*eV, Tcmb0=2.7255)

    #Set up of simulation parameters for example
    boxsize = (900 / 1000) / (H0 / 100.) # Convert 900 Mpc/h into Gpc
    n_particles = 832**3

    #Desired output set up
    nside = 128 #output maps will have nside=128, this must be lower or equal to the input nside of the simulation lightcone maps
    zi=0.0 #compute the contribution from z=0
    zf=1.05 #compute the contribution to the given signal until z~1 (this is as far as our simultions allow)

    #In this example we use "fast_mode" this should always be tested against the more robust default (fast_mode=False)- see comparison later

    IA = 0.0 #no instrinsic alignment in this example
    shift_nz = 0.0  #no delta z shift

    construct_class = construct_maps.construct_map_cosmogrid(maps=shells, z_low=lower_z, z_high=upper_z,
         nside=nside, boxsize=boxsize, cosmo=cosmo, n_particles=n_particles, zi=zi, zf=zf)

    kappa_wl_map_fast_mode = construct_class.construct_kappa_map(n_of_z=n_z, shift_nz=shift_nz, IA=IA, fast_mode=True, \
            fast_mode_num_points_1d=13, fast_mode_num_points_2d=512)
    
    cmb_kappa_map = construct_class.construct_kappa_cmb_map()

    hp.mollview(cmb_kappa_map, title=r"CMB $\kappa$ map", cmap='jet')
    plt.savefig("/capstor/scratch/cscs/damrein/outputs/plots/weak_lensing/cmb_kappa_map.png", dpi=300)




