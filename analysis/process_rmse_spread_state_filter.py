import numpy as np
import pickle
import glob
import ipdb

tanl = 0.05
nanl = 40000
burn = 5000
diffusion = 0

method_list = ['enkf', 'etkf', 'enks', 'etks', 'ienkf']
data = {
        'enkf_fore_rmse': np.zeros([21, 29]),
        'enkf_fore_spread': np.zeros([21, 29]),
        'etkf_fore_rmse': np.zeros([21, 29]),
        'etkf_fore_spread': np.zeros([21, 29]),
        'enks_fore_rmse': np.zeros([21, 29]),
        'enks_fore_spread': np.zeros([21, 29]),
        'etks_fore_rmse': np.zeros([21, 29]),
        'etks_fore_spread': np.zeros([21, 29]),
        'ienkf_fore_rmse': np.zeros([21, 29]),
        'ienkf_fore_spread': np.zeros([21, 29]),
        'enkf_anal_rmse': np.zeros([21, 29]),
        'enkf_anal_spread': np.zeros([21, 29]),
        'etkf_anal_rmse': np.zeros([21, 29]),
        'etkf_anal_spread': np.zeros([21, 29]),
        'enks_anal_rmse': np.zeros([21, 29]),
        'enks_anal_spread': np.zeros([21, 29]),
        'etks_anal_rmse': np.zeros([21, 29]),
        'etks_anal_spread': np.zeros([21, 29]),
        'ienkf_anal_rmse': np.zeros([21, 29]),
        'ienkf_anal_spread': np.zeros([21, 29]),
       }

def process_data(fnames):
    # loop columns
    for j in range(29):        
        #loop rows
        for i in range(21):
            f = open(fnames[i + j*21], 'rb')
            tmp = pickle.load(f)
            f.close()
            
            ana_rmse = tmp['anal_rmse']
            ana_spread = tmp['anal_spread']

            for_rmse = tmp['fore_rmse']
            for_spread = tmp['fore_spread']

            data[method + '_anal_rmse'][20 - i, j] = np.mean(ana_rmse[burn: nanl+burn])
            data[method + '_anal_spread'][20 - i, j] = np.mean(ana_spread[burn: nanl+burn])

            data[method + '_fore_rmse'][20 - i, j] = np.mean(for_rmse[burn: nanl+burn])
            data[method + '_fore_spread'][20 - i, j] = np.mean(for_spread[burn: nanl+burn])

for method in method_list:
    fnames = sorted(glob.glob('../filter_state_data/' + method + '/*diffusion_' + str(diffusion).zfill(3) + \
                              '*_nanl_' + str(nanl+burn) +  '_tanl_' + str(tanl) + '*' ))

    process_data(fnames)


f = open('./processed_rmse_spread_diffusion_' + str(diffusion).zfill(3) + '_nanl_' + str(nanl) +\
         '_tanl_' + str(tanl) + '_burn_' + str(burn) + '.txt', 'wb')

pickle.dump(data, f)
f.close()
