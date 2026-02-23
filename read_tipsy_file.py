import numpy as np

def read_tipsy(nBody_file_in, Lbox):
    try:
        f = open(nBody_file_in, 'r')
    except IOError:
        print('IOERROR: N-body tipsy file does not exist!')
        print('Define par.files.partfile_in = "/path/to/file"')
        exit()

    #header
    p_header_dt = np.dtype([('a','>d'),('npart','>i'),('ndim','>i'),('ng','>i'),('nd','>i'),('ns','>i'),('buffer','>i')])
    p_header = np.fromfile(f, dtype=p_header_dt, count=1, sep='')

    #particles
    p_dt = np.dtype([('mass','>f'),("x",'>f'),("y",'>f'),("z",'>f'),("vx",'>f'),("vy",'>f'),("vz",'>f'),("eps",'>f'),("phi",'>f')])
    p = np.fromfile(f, dtype=p_dt, count=int(p_header['npart']), sep='')

    #from tipsy units to [0,Lbox] in units of Lbox
    p['x']=Lbox*(p['x']+0.5)
    p['y']=Lbox*(p['y']+0.5)
    p['z']=Lbox*(p['z']+0.5)

    print('Reading tipsy-file done!')
    return p, p_header