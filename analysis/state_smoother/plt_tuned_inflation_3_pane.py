import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import Normalize
from matplotlib import rc
from matplotlib.colors import LogNorm
import glob
import ipdb
from matplotlib.colors import LogNorm
import h5py as h5

methods = ["ienks-bundle", "etks_hybrid", "etks_classic"]
tanl = 0.05
mda = 'false'
total_lag = 53
total_ens = 44

f = h5.File('./processed_all_smoother_state_diffusion_0.00_tanl_' + str(tanl).ljust(4,"0")+ '_nanl_20000_burn_05000_mda_' + mda + '.h5', 'r')


def find_optimal_inflation_values(method, data):
    tuning_stat = 'anal'
    tuned_rmse = np.array(f[method + '_' + tuning_stat + '_rmse'])
    tuned_rmse_min_vals = np.min(tuned_rmse, axis=1)
    lag, ens = np.shape(tuned_rmse_min_vals)
    inflation_range = np.linspace(1.0, 1.1, 11)[::-1]
    
    inflation_vals = np.zeros([lag, ens])

    for i in range(lag):
        for j in range(ens):
            min_val = tuned_rmse_min_vals[i,j]
            indx = tuned_rmse[i,:,j] == min_val

            inflation_vals[i,j] = inflation_range[indx]
   
    inflation_vals = np.transpose(inflation_vals)

    return inflation_vals 



fig = plt.figure()
ax0 = fig.add_axes([.935, .085, .020, .820])
ax1 = fig.add_axes([.070, .085, .280, .820])
ax2 = fig.add_axes([.360, .085, .280, .820])
ax3 = fig.add_axes([.650, .085, .280, .820])


color_map = sns.color_palette("viridis", 11)
max_scale = 1.00
min_scale = 1.10

ax_list = [ax1, ax2, ax3]

for i in range(0,3):
    inflation_vals = find_optimal_inflation_values(methods[i], f)
    sns.heatmap(inflation_vals, linewidth=0.5, ax=ax_list[i], cbar_ax=ax0, vmin=min_scale, vmax=max_scale, cmap=color_map)


x_labs = []
x_tics = []
x_vals = range(15, total_ens, 2)

for i in range(len(x_vals)):
    if i % 2 == 0:
        x_labs.append(str(x_vals[i]))
        x_tics.append(i)

y_labs = []
y_tics = []

y_vals = np.arange(1,total_lag, 3)
tic_vals = range(len(y_vals), 0, -1)
for i in range(len(y_vals)):
    if i % 3 == 0:
        y_labs.append(str(y_vals[i]))
        y_tics.append(tic_vals[i])
y_labs.append(str(y_vals[-1]))
y_tics.append(tic_vals[-1])


ax1.set_yticks(y_tics)
ax2.set_yticks(y_tics)
ax3.set_yticks(y_tics)
ax1.set_yticklabels(y_labs, va="bottom", rotation=0)

ax1.set_xticks(x_tics)
ax2.set_xticks(x_tics)
ax3.set_xticks(x_tics)
ax1.set_xticklabels(x_labs, rotation=0, ha="left")
ax2.set_xticklabels(x_labs, rotation=0, ha="left")
ax3.set_xticklabels(x_labs, rotation=0, ha="left")

ax0.tick_params(
        labelsize=20)

ax1.tick_params(
        labelsize=20)

ax2.tick_params(
        labelleft=False,
        labelsize=20)

ax3.tick_params(
        labelleft=False,
        labelsize=20)

if mda=="true":
    fig_title = "Tuned inflation values, MDA, shift 1, $\Delta$t="+ str(tanl)

else:
    fig_title = "Tuned inflation values, shift 1, $\Delta$t="+ str(tanl)

plt.figtext(.080, .905, "IEnKS Bundle", horizontalalignment='left', verticalalignment='bottom', fontsize=20)
plt.figtext(.500, .905, "ETKS Hybrid", horizontalalignment='center', verticalalignment='bottom', fontsize=20)
plt.figtext(.920, .905, "ETKS Classic", horizontalalignment='right', verticalalignment='bottom', fontsize=20)

plt.figtext(.025, .52, r'Lag length', horizontalalignment='center', verticalalignment='center', fontsize=24, rotation='90')
plt.figtext(.50, .02, r'Ensemble size', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.5, .97, fig_title,
        horizontalalignment='center', verticalalignment='center', fontsize=24)


plt.show()
