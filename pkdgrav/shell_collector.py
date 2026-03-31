import multiprocessing as mp
import healpy as hp
import numpy as np
import argparse
import sys
import os
import re

"""
This script collects the lightcone output and renames the files
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collects the lightcone output of pkdgrav to shells.')
    parser.add_argument('--param_file', type=str, help='Path to the param file used to start the sim', required=True)

    # parse
    args = parser.parse_args()
    print("Collecting PkDgrav3 output to shells...", flush=True)

    # get all the necessary stuff from the param file
    with open(args.param_file, "r") as f:
        content = f.read()

    # some regex stuff
    print(f"Extracting params from: ", args.param_file, flush=True)
    namespace = re.search('(?<=achOutName)\s*=\s*["\'].+["\']', content)[0].split("=")[-1].strip().replace('"', '').replace("'", "")
    print(f"achOutName:   ", namespace, flush=True)
    nside = int(re.search("(?<=nSideHealpix)\s*=\s*\d+", content)[0].split("=")[-1].strip())
    print(f"nSideHealpix: ", nside, flush=True)
    z_out = float(re.search("(?<=dRedshiftLCP)\s*=\s*\d+\.\d+", content)[0].split("=")[-1].strip())
    print(f"dRedshiftLCP: ", z_out, flush=True)
    steps = int(re.search("(?<=nSteps)\s*=\s*\d+", content)[0].split("=")[-1].strip())
    print(f"nSteps:       ", steps, flush=True)
    
    # get the number of pixels per map
    npix = hp.nside2npix(nside)

    # read out the log file
    z_output = np.genfromtxt(namespace + ".log")[:, 1]


    def process_single(n_step):
        current_pixels = []
        current_proc = 0
        while True:
            # if we expect everything to be empty just remove the files
            if z_output[n_step] > z_out:
                break

            # collect and rename
            else:
                # get the current file
                current_file = os.path.join(namespace + ".%05d.hpb.%i" % (n_step, current_proc))
                if os.path.exists(current_file):
                    """
                    The output in the hpb file (see hpb2fits.c in pkdgrav3 code) is ordered in the RING scheme.
                    each file contains the grouped parts, ungrouped parts and the potential
                    for our application we only need the ungouped parts (uint32) starting from 1
                    the last file is padded with zeros in case the number of processors does not divide
                    the number of pixels
                    """
                    data = np.fromfile(current_file, count=-1, dtype=np.uint32)
                    # add group parts to the other (from tools in pkdgrav3)
                    #assert np.sum(data[0::3]) == 0, "There are apparently grouped parts..."
                    current_pixels.append(data[0::3] + data[1::3])
                    current_proc += 1
                else:
                    # concat and truncate
                    current_pixels = np.concatenate(current_pixels)
                    # assert is still a statement in python 3 -> no parantheses
                    assert np.sum(current_pixels[npix:]) == 0, "Truncation did not work (pkd_collect)"
                    current_pixels = current_pixels[:npix]

                    # save the file with the correct naming scheme
                    z_0 = z_output[n_step]

                    # catch very small z
                    if np.abs(z_0) <= 1e-9:
                        z_0 = 0.0

                    z_1 = z_output[n_step - 1]

                    # catch larger than output redshifts
                    if z_1 > z_out:
                        z_1 = z_out

                    # save
                    file_name = f"{namespace}-shell_z-high={z_1}_z-low={z_0}.fits"
                    hp.write_map(file_name, m=current_pixels, nest=False, dtype=np.float32)

                    sys.stdout.write("Collected: {} \n".format(file_name))
                    sys.stdout.flush()
                    # break the loop
                    break

        # loop to remove files (only afer master file is written)
        current_proc = 0
        while True:
            current_file = os.path.join(namespace + ".%05d.hpb.%i" % (n_step, current_proc))
            if os.path.exists(current_file):
                os.remove(current_file)
                current_proc += 1
            else:
                break

        return 0

    pool = mp.Pool(processes=12)
    multi_collect = [pool.apply_async(process_single,
                     args=(n_step,)) for n_step in range(1, steps + 1)]
    out = np.array([p.get() for p in multi_collect])
    pool.close()

