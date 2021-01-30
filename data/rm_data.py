import glob
import os

fnames = glob.glob("etks_classic/*")
for name in fnames:
        os.system('rm ' + name)

