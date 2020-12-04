import matplotlib.pyplot as plt
plt.rcdefaults()
import numpy as np
import math
from collections import defaultdict
from scipy.optimize import curve_fit
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pickle
import os.path
from numpy import linalg as LA


def read_warray(method, ky, dir_name):
    with open('./{}/long_multi_Akw_t120/{}_{}_dict.txt'.format(dir_name, method, ky),'r') as f:
        lines = f.readlines()
        warray = lines[0]
        warray = [np.real(np.complex(num)) for num in warray.split()]

        wvalue_array = []
        for i in range(len(lines[1:])):
            if i % 2 == 0:
                kval = float(lines[i+1].strip())
            else:
                wvalue = [np.complex(num) for num in lines[i+1].split()]
                wvalue_array.append(wvalue)
        wvalue_array = np.array(wvalue_array)
    return warray, wvalue_array

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

U = 8
result_dir = '8th'
ky = 'ky0'
method = 'recursion'
# method = 'recursion_lp'
# method = 'lp'

for result_dir in ['half']:
    for U in [8]:
        legend_list = ['DMRG','Recursion']
        clist = ['b','r','g']
        idx = 0
        plt.figure(figsize=(12,9))
        for method in ['recursion']:
            ky0_warray_dict = dict()
            ky0_density_dict = dict()
            ky1_warray_dict = dict()
            ky1_density_dict = dict()
            for ky in ['ky0','ky1']:
                for dag_opt in ['_dag','']:
                    dir_name = '{}_results/U{}{}'.format(result_dir, U, dag_opt)
                    print(dir_name)
                    warray, wvalue_array = read_warray(method, ky, dir_name)
                    print(wvalue_array.shape)
                    print(len(warray))

                    wvalue_array = np.real(wvalue_array)
                    print(wvalue_array.shape)
                    sum_k = np.sum(wvalue_array,axis=0)
                    print(sum_k.shape)

                    wvalue_array = np.transpose(wvalue_array)
                    vl, wl = np.shape(wvalue_array)
                    xp = np.linspace(0, 1, wl)
                    xvals = np.linspace(0, 1, wl*4)
                    inter_wvalue_array = []
                    density_list = []
                    
                    for i in range(len(wvalue_array)):
                        temp_wval = np.interp(xvals, xp, wvalue_array[i])
                        inter_wvalue_array.append(temp_wval)  
                        density_list.append( np.sum(wvalue_array[i])/len(wvalue_array[i]) )
                    
                    density_list = np.array(density_list)
                    # plt.plot(warray, density_list)
                    
                    wvalue_array = np.array(inter_wvalue_array)
                    vl, wl = np.shape(wvalue_array)

                    if ky == 'ky0':
                        if dag_opt == '_dag':
                            ky0_warray_dict['dag'] = wvalue_array
                            ky0_density_dict['dag'] = density_list
                        else:
                            ky0_warray_dict['nodag'] = wvalue_array
                            ky0_density_dict['nodag'] = density_list
                    else:
                        if dag_opt == '_dag':
                            ky1_warray_dict['dag'] = wvalue_array
                            ky1_density_dict['dag'] = density_list
                        else:
                            ky1_warray_dict['nodag'] = wvalue_array
                            ky1_density_dict['nodag'] = density_list
                    wvalue_dict = dict()
                    for i in range(len(warray)):
                        kval = int(warray[i]*100)
                        wvalue_dict[kval] = wvalue_array[i]

                    wgrid_list = sorted(list(wvalue_dict.keys()))
                    print(wgrid_list[0], wgrid_list[-1])

            warray = np.array(warray)
            if result_dir == 'half':
                warray = warray - U/2.0    

            warray_sum = warray[-1]-warray[0]

            density_all = (ky0_density_dict['dag']+ky0_density_dict['nodag']) + (ky1_density_dict['dag']+ky1_density_dict['nodag'])

            warray_tail = []
            density_tail_val = []
            wl_idx = 0
            for wl_idx in range(len(warray)):
                if warray[wl_idx] < -warray[-1]:
                    print(wl_idx, warray[wl_idx])
                    warray_tail.append(-1.0*warray[wl_idx])
                    density_tail_val.append(density_all[wl_idx])

            warray = np.concatenate([warray, warray_tail[::-1]])
            density_all = np.concatenate([density_all, density_tail_val[::-1]])
            ky0_density_dict['dag'] = np.concatenate([ky0_density_dict['dag'], density_tail_val[::-1]])
            ky0_density_dict['nodag'] = np.concatenate([ky0_density_dict['nodag'], density_tail_val[::-1]])
            ky1_density_dict['dag'] = np.concatenate([ky1_density_dict['dag'], density_tail_val[::-1]])
            ky1_density_dict['nodag'] = np.concatenate([ky1_density_dict['nodag'], density_tail_val[::-1]])

            norm_factor = np.sum(density_all)*warray_sum/len(warray)

            print('norm_factor: ',norm_factor)
            density_all = density_all/norm_factor
            # density_all = density_all/(2*np.pi)

            plt.plot(warray, density_all,c=clist[idx],label=legend_list[idx])
            
            plt.legend(fontsize=20)
            plt.ylabel('$N(\omega)$',fontsize=24)
            if result_dir == 'half':
                plt.xlabel('$\omega-U/2$',fontsize=24)
            else:
                plt.xlabel('$\omega$',fontsize=24)

            plt.axvline(x=0,c='g')
            plt.tick_params(direction='in',labelsize=20)

            if method == 'recursion':
                a = plt.axes([0.125, .5, .3, .3])
                plt.plot(warray, (ky0_density_dict['nodag']),label='$C,ky=0$')
                plt.plot(warray, (ky1_density_dict['nodag']),label='$C,ky=\pi$')
                plt.plot(warray, (ky0_density_dict['dag']),label='$C^{\dagger},ky=0$')
                plt.plot(warray, (ky1_density_dict['dag']),label='$C^{\dagger},ky=\pi$')
                plt.text(-15, 0.12, '$N(k,\omega)$',fontsize=20)
                
                plt.tick_params(
                    axis='y',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    left=False,         # ticks along the top edge are off
                    labelleft=False) # labels along the bottom edge are off
                
            idx+=1  

        plt.legend(fontsize=12)
        plt.tick_params(direction='in',labelsize=12)
        plt.show()