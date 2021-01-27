import numpy as np
import pickle
import glob
import ipdb

tanl = 0.05
nanl = 40000
burn = 5000
#diffusion = 0
shift_equal_lag = False


method_list = ['etks', 'enks']
data = {
        'enks_fore_rmse': np.zeros([11, 28, 21]),
        'enks_fore_spread': np.zeros([11, 28, 21]),
        'enks_filter_rmse': np.zeros([11, 28, 21]),
        'enks_filter_spread': np.zeros([11, 28, 21]),
        'enks_smooth_rmse': np.zeros([11, 28, 21]),
        'enks_smooth_spread': np.zeros([11, 28, 21]),
        'etks_fore_rmse': np.zeros([11, 28, 21]),
        'etks_fore_spread': np.zeros([11, 28, 21]),
        'etks_filter_rmse': np.zeros([11, 28, 21]),
        'etks_filter_spread': np.zeros([11, 28, 21]),
        'etks_smooth_rmse': np.zeros([11, 28, 21]),
        'etks_smooth_spread': np.zeros([11, 28, 21]),
       }

def process_data(fnames, shift_equal_lag):
    indx = []
    exps = []


    for i in range(len(fnames)):
        indx.append(True)

    for i in range(len(fnames)):
        name = fnames[i]
        split_name = name.split('_')
        lag = int(split_name[-9])
        shift = int(split_name[-7])

        if lag > 1:

            if shift_equal_lag:
                if shift==lag:
                    exps.append(name)
                
            else:
                if shift!=lag:
                    exps.append(name)

        else:
            exps.append(name)

    # outter loop in lag value
    for k in range(11):
        # second loop over ensemble size
        for j  in range(28):
            # inner loop over the inflation values
            for i in range(21):
                
                name = exps[i + j * 21 + k * 21 * 28]
                f = open(name,'rb')
                tmp = pickle.load(f)
                f.close()

                ana_rmse = tmp['anal_rmse']
                ana_spread = tmp['anal_spread']

                fil_rmse = tmp['filt_rmse']
                fil_spread = tmp['filt_spread']

                for_rmse = tmp['fore_rmse']
                for_spread = tmp['fore_spread']

                data[method + '_smooth_rmse'][10 - k, j, 20 - i] = np.mean(ana_rmse[burn: nanl+burn])
                data[method + '_smooth_spread'][10 - k, j, 20 - i] = np.mean(ana_spread[burn: nanl+burn])

                data[method + '_filter_rmse'][10 - k, j, 20 - i] = np.mean(fil_rmse[burn: nanl+burn])
                data[method + '_filter_spread'][10 - k, j, 20 - i] = np.mean(fil_spread[burn: nanl+burn])

                data[method + '_fore_rmse'][10 - k, j, 20 - i] = np.mean(for_rmse[burn: nanl+burn])
                data[method + '_fore_spread'][10 - k, j, 20 - i] = np.mean(for_spread[burn: nanl+burn])

    smooth_rmse = np.amin(data[method + '_smooth_rmse'], axis=2)
    smooth_spread = np.zeros(np.shape(smooth_rmse))

    filter_rmse = np.zeros(np.shape(smooth_rmse)) 
    filter_spread = np.zeros(np.shape(smooth_rmse))

    fore_rmse = np.zeros(np.shape(smooth_rmse)) 
    fore_spread = np.zeros(np.shape(smooth_rmse))

    # outter loop in lag value
    for k in range(11):
        # second loop over ensemble size
        for j  in range(28):
            indx_smooth = np.where(data[method + '_smooth_rmse'][k, j, :] == smooth_rmse[k, j])
            smooth_spread[k, j] = data[method + '_smooth_spread'][k, j, indx_smooth[0]]
            
            filter_rmse[k, j] = data[method + '_filter_rmse'][k, j, indx_smooth[0]]
            filter_spread[k, j] = data[method + '_filter_spread'][k, j, indx_smooth[0]]

            fore_rmse[k, j] = data[method + '_fore_rmse'][k, j, indx_smooth[0]]
            fore_spread[k, j] = data[method + '_fore_spread'][k, j, indx_smooth[0]]


    data[method + '_smooth_rmse'] = smooth_rmse
    data[method + '_smooth_spread'] = smooth_spread 
    data[method + '_filter_rmse'] = filter_rmse
    data[method + '_filter_spread'] = filter_spread 
    data[method + '_fore_rmse'] = fore_rmse
    data[method + '_fore_spread'] = fore_spread 


for method in method_list:
    fnames = sorted(glob.glob('../smoother_state_data/' + method + '/*' ))

    process_data(fnames, shift_equal_lag)


f = open('./processed_shift_equal_lag_' + str(shift_equal_lag) + '_smoother_state_rmse_spread_nanl_' + str(nanl) +\
         '_tanl_' + str(tanl) + '_burn_' + str(burn) + '.txt', 'wb')

pickle.dump(data, f)
f.close()
