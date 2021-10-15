import glob
import os

fnames = glob.glob("enks_hybrid/*")
for name in fnames:
        os.system('rm ' + name)

