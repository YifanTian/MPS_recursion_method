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
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pickle
import os.path
from numpy import linalg as LA
from scipy.interpolate import interp1d


def read_warray(method, ky, dir_name, method_dir):
    idx = 0
    with open('./{}/{}/{}_{}_dict.txt'.format(dir_name, method_dir, method, ky),'r') as f:
    # with open('./{}/Akw/recursion_lp_ky1_dict.txt'.format(dir_name),'r') as f:
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

U = 4
result_dir = '8th'
ky = 'ky0'
method = 'recursion'
# method = 'recursion_lp'
# method = 'lp'

viridis = cm.get_cmap('gist_ncar_r', 256)
newcolors = viridis(np.linspace(0, 1, 256))
white = np.array([255/256, 255/256, 255/256, 1])
fst = white     # white
snd = newcolors[15]                # pink

linfit = interp1d([1,15], np.vstack([fst, snd]), axis=0)
nclist = linfit([i+1 for i in range(15)])
newcolors = np.concatenate([nclist, newcolors[15:,:]])
newcmp = ListedColormap(newcolors)

cp_dict = dict()
cp_dict['half'] = {1:0.5,2:1,4:2,8:4}
cp_dict['4th'] = {1:((-156.07769-(-155.0020))/4),2:((-142.0900-(-142.23666))/4),4:((-121.38045-(-123.47701))/4),8:((-99.8226-(-103.905940))/4)}
cp_dict['8th'] = {1:((-156.84497-(-157.26119))/4),2:((-137.1301-(-139.03185))/4),4:((-107.737785-(-111.9965))/4),8:((-77.91235-(-84.15223))/4)}

minimum = 1.0 
maximum = 0.0
label_dict = dict(zip([0,1,2,3,4,5,6,7],['a','b','c','d','e','f','g','h']))
# fill_number_dict = {'half':1,'8th':0.875,'4th':0.75}
# for result_dir in ['half','8th','4th']:
# # for U in [1,2,4,8]:
# for result_dir in ['4th','half','8th']:
for result_dir in ['half']:
    l_idx = 0
    fig, axs = plt.subplots(2, 4, sharey=True, figsize=(14,10))
    for f_idx, U in zip([0,1,2,3],[1,2,4,8]):
        chemical_p = cp_dict[result_dir][U]
        label_shift = 0
        fig.subplots_adjust(wspace=0.15,hspace=0.2)
        for ky_idx,ky in zip([0,1],['ky0','ky1']):
            legend_list = ['DMRG','Recursion']
            clist = ['g','r','b']
            idx = 0
            for method in ['recursion']:
                warray_dict = dict()
                density_dict = dict()
                for dag_opt in ['_dag','']:
                    dir_name = '{}_results/U{}{}'.format(result_dir, U, dag_opt)
                    print(dir_name)
                    method_dir = 'long_multi_delta_re_matrix_Akw_t100_tlimit500_factor2'
                    warray, wvalue_array = read_warray(method, ky, dir_name, method_dir)
                    print(wvalue_array.shape)
                    print(len(warray))

                    wvalue_array = np.real(wvalue_array)
                    wvalue_array = np.transpose(wvalue_array)
                    vl, wl = np.shape(wvalue_array)
                    xp = np.linspace(0, 1, wl)
                    xvals = np.linspace(0, 1, wl*4)
                    inter_wvalue_array = []
                    density_list = []
                    
                    for i in range(len(wvalue_array)):
                        temp_wval = np.interp(xvals, xp, wvalue_array[i])
                        inter_wvalue_array.append(temp_wval)  
                        density_list.append(np.sum(wvalue_array[i])/len(wvalue_array[i]))
                    
                    density_list = np.array(density_list)
                    
                    wvalue_array = np.array(inter_wvalue_array)
                    vl, wl = np.shape(wvalue_array)

                    if dag_opt == '_dag':
                        warray_dict['dag'] = wvalue_array
                        density_dict['dag'] = density_list
                    else:
                        warray_dict['nodag'] = wvalue_array
                        density_dict['nodag'] = density_list

                    # wvalue_array = wvalue_array[::-1,:]
                    wvalue_dict = dict()
                    for i in range(len(warray)):
                        # print(int(warray[i]*100)/100.0)
                        kval = int(warray[i]*100)
                        wvalue_dict[kval] = wvalue_array[i]

                    wgrid_list = sorted(list(wvalue_dict.keys()))
                    # print(wgrid_list[0]/100.0,wgrid_list[-1]/100.0)
                    print(wgrid_list[0],wgrid_list[-1])

                print(warray_dict['dag'].shape)
                print(warray_dict['nodag'].shape)
                warray_all = 1.0*(warray_dict['dag']+warray_dict['nodag'])

                warray_sum = warray[-1]-warray[0]
                warray_norm_factor = np.sum(warray_all)*warray_sum*np.pi/warray_all.shape[0]/warray_all.shape[1]
                warray_all = warray_all/warray_norm_factor

                density_all = 1.0*(density_dict['dag']+density_dict['nodag'])
                norm_factor = np.sum(density_all)*warray_sum/len(warray)
                print('norm_factor: ',norm_factor)
                density_all = density_all/norm_factor

                if U == 1:
                    if ky == 'ky0':
                        axs[ky_idx,f_idx].set_ylabel('$\omega(k_y=0)$',fontsize=16)
                    else:
                        axs[ky_idx,f_idx].set_ylabel('$\omega(k_y=\pi)$',fontsize=16)

                warray_all[warray_all<0.0001] = 0.0
                
                if result_dir == 'half':
                    print('ky_idx: ',ky_idx)
                    axs[ky_idx,f_idx].set_ylim([-9-label_shift,9-label_shift])
                    if ky == 'ky0':
                        axs[ky_idx,f_idx].text(-0.9, 7.5-label_shift, r'$({})\ U={}$'.format(label_dict[l_idx],U), fontdict=font)
                        cax = axs[ky_idx,f_idx].imshow(warray_all[::-1,:], aspect='auto', extent=[-1,1,wgrid_list[0]/100.0-chemical_p,wgrid_list[-1]/100.0-chemical_p],cmap=newcmp)
                        l_idx += 1
                    else:
                        axs[ky_idx,f_idx].text(0.1, 7.5-label_shift, r'$({})\ U={}$'.format(label_dict[3+l_idx],U), fontdict=font)
                        new_warray_all = warray_all[::-1,:]
                        h,w = new_warray_all.shape
                        new_warray_all = np.concatenate([new_warray_all[:,int(w/2):],new_warray_all[:,0:int(w/2)]],axis=1)
                        cax = axs[ky_idx,f_idx].imshow(new_warray_all, aspect='auto', extent=[0,2,wgrid_list[0]/100.0-chemical_p,wgrid_list[-1]/100.0-chemical_p],cmap=newcmp)
                        axs[ky_idx,f_idx].set_xlabel('$k_x/\pi$',fontsize=16)
                    axs[ky_idx,f_idx].axhline(y=0,c='black',ls='--',linewidth=0.8)
                else:
                    axs[ky_idx,f_idx].set_ylim([-6-label_shift,12-label_shift])
                    if result_dir == '4th':
                        dir_name = '1/4'
                    else:
                        dir_name = '1/8'
                    if ky == 'ky0':             
                        axs[ky_idx,f_idx].text(-0.9, 10.5, r'$({})\ U={}$'.format(label_dict[l_idx], U), fontdict=font)
                        cax = axs[ky_idx,f_idx].imshow(warray_all[::-1,:], aspect='auto', extent=[-1,1,wgrid_list[0]/100.0-chemical_p,wgrid_list[-1]/100.0-chemical_p],cmap=newcmp)
                        l_idx += 1
                    else:
                        axs[ky_idx,f_idx].text(0.1, 10.5, r'$({})\ U={}$'.format(label_dict[3+l_idx], U), fontdict=font)

                        new_warray_all = warray_all[::-1,:]
                        h,w = new_warray_all.shape
                        new_warray_all = np.concatenate([new_warray_all[:,int(w/2):],new_warray_all[:,0:int(w/2)]],axis=1)
                        cax = axs[ky_idx,f_idx].imshow(new_warray_all, aspect='auto', extent=[0,2,wgrid_list[0]/100.0-chemical_p,wgrid_list[-1]/100.0-chemical_p],cmap=newcmp)
                        axs[ky_idx,f_idx].set_xlabel('$k_x/\pi$',fontsize=16)

                    axs[ky_idx,f_idx].axhline(y=0,c='black',ls='--',linewidth=0.8)
                if np.max(warray_all) > maximum:
                    global_cax = cax
                    maximum = np.max(warray_all)
                idx+=1
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.83, 0.15, 0.015, 0.7])
    fig.colorbar(global_cax, cax=cbar_ax)
    plt.show()

