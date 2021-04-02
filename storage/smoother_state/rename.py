import glob
import os
import ipdb

fnames = glob.glob("enks-n-dual_classic/diffusion_0.00/*")
for name in fnames:
    split_name = name.split("_")
    tmp = split_name[:2] + [split_name[2] + "-dual"] +  split_name[3:]
    rename = ""
    for i in range(len(tmp)-1):
        rename += tmp[i] + "_"

    rename += tmp[-1]
    #ipdb.set_trace()
    cmd = 'mv ' + name + " " + rename
    os.system(cmd)
