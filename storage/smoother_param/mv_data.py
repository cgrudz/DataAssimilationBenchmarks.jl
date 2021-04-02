import glob
import os
import ipdb

fnames = glob.glob("/x/capc/cgrudzien/da_benchmark/data/enks_classic/*")
for name in fnames:
    os.system("mv " + name + " enks_classic/diffusion_0.00/" )

