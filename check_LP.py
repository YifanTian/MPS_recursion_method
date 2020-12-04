import numpy as np 
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy.linalg import inv
import matplotlib.animation as animation

import spectrum
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def fourier_transform(sample):
    n = len(sample)
    N = 2*n
    Aw = np.zeros(2*N,dtype=np.complex128)
    warray = np.zeros(2*N,dtype=np.complex128)
    for wn in range(2*N):
        for x in range(n):
            w = 2*np.pi*wn/N
            warray[wn] = w
            t = (x-int(n/2))*0.1                        # right
            Aw[wn] += 0.1*sample[x]*np.exp(1j*w*t)
    return warray, Aw

def complete_evo(occilation):
    occilation = np.array(occilation)
    occilation = np.append(np.conj(np.flip(occilation[1:],0)),occilation)
    return occilation

def burg_interpolation(data, order, extend_L, method='arcovar'):
    N = order
    M = len(data)
    P = len(data)+extend_L
    t = list(range(len(data)))
    [a, E, K] = spectrum.burg.arburg(data, N)
    a = np.hstack((1,a))

    y = np.zeros(len(data)+extend_L,dtype=complex)
    x_pred = list(range(len(y)))
    y[0:M] = data[0:M]
    # in reality, you would use `filter` instead of the for-loop
    for ii in range(M,P):    
        y[ii] = -sum(a[1:] * y[(ii-1):(ii-N-1):-1] )
    return y


def k0_frequency(dir_name, N, time_limit_list, ticker_list, show=True):
    idx = 0
    for time_limit in time_limit_list:
        img_occilation = []
        real_occilation = []
        tarray = []
        for i in range(0,time_limit):    
            t = i*0.1
            phase = math.fmod(2.81357*i,2*np.pi)
            try:
                filename = dir_name + 'Green_t%.3f.txt' % t
                print(filename)
                sz_sites, img_profile = read_lines(filename, "Green_img")
                sz_sites, real_profile = read_lines(filename, "Green_real")

                real_profile = np.array(real_profile)
                img_profile = np.array(img_profile)

                img_occilation.append(img_profile[int(len(img_profile)/2)])
                real_occilation.append(real_profile[int(len(real_profile)/2)])
                tarray.append(t)
                print(filename)
            except Exception as e:
                print(e)

        print(len(tarray))
        maxT = tarray[-1]
        tarray = np.array(tarray)
        real_occilation = real_occilation*np.exp(-8*(tarray/maxT)**2)
        img_occilation = img_occilation*np.exp(-8*(tarray/maxT)**2)
        tarray = np.append(-1*np.flip(tarray[1:],0),tarray)
        real_occilation = np.array(real_occilation)
        img_occilation = np.array(img_occilation)
        real_occilation = np.append(np.flip(real_occilation[1:],0),real_occilation)
        img_occilation = np.append(-1*np.flip(img_occilation[1:],0),img_occilation)

        warray, w_value = fourier_transform(real_occilation+1j*img_occilation)


        w_value = np.array(w_value)/np.pi
        if show:
            # plt.xlim((0,8))
            plt.xlabel('$\omega$',fontsize=16)
            plt.ylabel('$N(\omega)$',fontsize=16)
            print(idx, ticker_list[idx],time_limit_list[idx])
            plt.plot(warray, w_value, ticker_list[idx], label='t={}'.format(time_limit_list[idx]*0.1))
            # plt.plot(warray, w_value)

        idx+=1
    plt.legend()
    plt.show()  

    wdensity = np.stack([np.array(warray),np.array(w_value)])
    np.save(dir_name+'warray', wdensity)
    return


def approximate_matrix(mm, N):
    center_signal = mm[int(len(mm)/2),:]
    size = len(mm)
    assert len(mm) == len(mm[0,:])
    new_mm = np.zeros((N, N), dtype=complex)
    val_dict = dict()
    for i in range(int(size/2)+1):
        val_dict[abs(i-int(len(mm)/2))] = center_signal[i]
    for i in range(N):
        for j in range(N):
            if abs(i-j) in val_dict:
                new_mm[i,j] = val_dict[abs(i-j)]
    
    return new_mm

def approximate_matrix_odd_even(mm, N):
    size = len(mm)
    assert len(mm) == len(mm[0,:])
    # new_mm = np.zeros((N, N), dtype=complex)
    new_mm = mm.copy()
    val_dict = dict()
    even_val_dict = dict()
    odd_val_dict = dict()

    # print(int(len(mm)/2),(int(len(mm)/2)-1))
    # assert int(len(mm)/2) % 2 == 0
    for i in range(int(size/2)+1):
        even_val_dict[abs(i-int(len(mm)/2))] = mm[int(len(mm)/2),:][i]
    # assert (int(len(mm)/2)-1) % 2 == 1
    for i in range(int(size/2)+1):
        odd_val_dict[abs(i-int(len(mm)/2)+1)] = mm[int(len(mm)/2)-1,:][i]

    for i in range(N):
    # for i in range(int(N/2)-40,int(N/2)+40):
        for j in range(N):
            if i%2 == 0:
                if abs(i-j) in even_val_dict:
                    new_mm[i,j] = even_val_dict[abs(i-j)]
            else:
                if abs(i-j) in odd_val_dict:
                    new_mm[i,j] = odd_val_dict[abs(i-j)]
    
    return new_mm

def lpc(y, m):
    "Return m linear predictive coefficients for sequence y using Levinson-Durbin prediction algorithm"
    #step 1: compute autoregression coefficients R_0, ..., R_m
    R = [y.dot(y)] 
    if R[0] == 0:
        return [1] + [0] * (m-2) + [-1]
    else:
        for i in range(1, m + 1):
            r = y[i:].dot(y[:-i])
            R.append(r)
        R = np.array(R)
    #step 2: 
        A = np.array([1, -R[1] / R[0]])
        E = R[0] + R[1] * A[1]
        for k in range(1, m):
            if (E == 0):
                E = 10e-17
            alpha = - A[:k+1].dot(R[k+1:0:-1]) / E
            A = np.hstack([A,0])
            A = A + alpha * A[::-1]
            E *= (1 - alpha**2)
        return A


def get_Green_snapshot_at_center(dirname,t,N):
    Green_snapshot = np.zeros((N),dtype=complex)
    file_name = './{}/Green_t{:.3f}.txt'.format(dirname,t)
    with open(file_name) as f:
        line = [0 for _ in range(N)]
        j = 0
        for e in f:
            nums = e.strip().split()
            n = float(nums[0]) + 1j*float(nums[1])
            line[j] = n
            j+=1 
        Green_snapshot[:] = line 
    return Green_snapshot


N = 100
def read_overlap(dir_name):
    overlap = np.zeros((N,N),dtype=np.complex_)
    for i in range(N):
        line = [0 for _ in range(N)]
        with open('./{}/Green/overlap/overlap{}.txt'.format(dir_name, i+1)) as f:
            j = 0
            for e in f:
                nums = e.strip().split()
                n = float(nums[1]) + 1j*float(nums[2])
                line[j] = n
                j+=1 
        overlap[i,:] = line 
    return overlap


def interpolation(data, order, extend_L):
    N = order
    M = len(data)
    P = len(data)+extend_L
    t = list(range(len(data)))
    a = lpc(data, N)

    y = np.zeros(len(data)+extend_L)
    x_pred = list(range(len(y)))
    y[0:M] = data[0:M]
    # in reality, you would use `filter` instead of the for-loop
    for ii in range(M,P):    
        y[ii] = -sum(a[1:-1] * y[(ii-1):(ii-N):-1] )
    return y

def get_2d_translational_green(green, large_N):
    N = len(green)
    even_green = green
    odd_green = green[::-1]
    even_green_dict = dict()
    odd_green_dict = dict()
    for i in range(N):
        even_green_dict[i-(int(N/2)-1)] = even_green[i]
    for i in range(N):
        odd_green_dict[i-int(N/2)] = odd_green[i]
    N = len(green)*1
    Green_temp = np.zeros((large_N,large_N),dtype=complex)
    for i in range(1,large_N,2):
        for j in range(large_N):
            if j - i in even_green_dict:
                Green_temp[i,j] = even_green_dict[j-i]
    for i in range(0,large_N,2):
        for j in range(large_N):
            if j - i in odd_green_dict:
                Green_temp[i,j] = odd_green_dict[j-i]
    return Green_temp


def get_ladder_translational_green(green,center_idx,path_to_ladder, large_N):
    center_idx = path_to_ladder[center_idx]
    green_dict = dict()
    original_green_dict = dict()
    Green_temp = np.zeros((large_N,large_N),dtype=complex)
    N = len(green)
    index_dict = dict()
    for i in range(N):
        xy_idx = path_to_ladder[i]
        relative_idx = (xy_idx[0] - center_idx[0],xy_idx[1] - center_idx[1])
        abs_relative_idx = (abs(relative_idx[0]),abs(relative_idx[1]))
        original_green_dict[relative_idx] = green[i]
        if abs_relative_idx not in green_dict:
            index_dict[abs_relative_idx] = [relative_idx]
            green_dict[abs_relative_idx] = green[i]
        else:
            green_dict[abs_relative_idx] = 0.5*(green[i]+green_dict[abs_relative_idx])
            index_dict[abs_relative_idx].append(relative_idx)

    try:
        for i in range(large_N):
            for j in range(large_N):
                new_center_idx = path_to_ladder[i]
                xy_idx = path_to_ladder[j]
                relative_idx = (xy_idx[0] - new_center_idx[0],xy_idx[1] - new_center_idx[1])
                relative_idx = (abs(relative_idx[0]),abs(relative_idx[1]))
                if relative_idx in green_dict:
                    Green_temp[i,j] = green_dict[relative_idx]
    except:
        print(relative_idx,  green_dict[relative_idx])
    return Green_temp


tlimit_compare = 150
tlimit = 240

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--doping', type=str,
                     help='doping',default='4th')
parser.add_argument('--start_time', type=int,
                     help='start_time',default=100)
parser.add_argument('--U', type=int,
                     help='U',default=8)
args = parser.parse_args()

N = 128
path_to_ladder = dict()
ix = 0
iy = 0
leg0_sites = []
leg1_sites = []
for n in range(N*4):
    if iy == 0:
        leg0_sites.append(n)
    else:
        leg1_sites.append(n)
    path_to_ladder[n] = (ix,iy)
    if n%2 == 0:
        iy = 1-iy
    else:
        ix+=1
    
evolve_facotr = 4
for doping in ['4th']:
    files = '../ladder_system_test/{}_results'.format(doping)
    new_dir_name = files.split('/')[-1]
    # for case in ['U{}'.format(args.U),'U{}_dag'.format(args.U)]:
    # for U in [1,2,4,8]:
    for U in [8]:
        for case in ['U{}'.format(U)]:
        # for case in ['U{}_dag'.format(U)]:
            file_name = files+'/'+case+'/'
            muti_num = 6
            interval_step = 5
            start_time = args.start_time
            muti_num_grid = [10]
            interval_step_grid = [5]

            res_grid = np.zeros([10, len(interval_step_grid)])
            dis_grid = np.zeros([10, len(interval_step_grid)])
            res_step_list = []
            dis_step_list = []
            m_idx = 0

            for m_idx, muti_num in enumerate(muti_num_grid):
                for i_idx, interval_step in enumerate(interval_step_grid):
                    recursion_step = start_time - (muti_num-1)*interval_step
                    if recursion_step <= 0:
                        res_grid[m_idx, i_idx] = res_grid[0, 0]
                        dis_grid[m_idx, i_idx] = dis_grid[0, 0]
                        continue

                    tlimit_compare = 150
                    tlimit = start_time*evolve_facotr
                    # tlimit = 300
                    extend_N = 0
                    Nreal = 128
                    factor = 4
                    large_GN = factor*Nreal
                    large_N = factor*Nreal
                    
                    overlap_Green = np.zeros((tlimit,large_N,large_N),dtype=complex)
                    Green_tn = np.zeros((tlimit,Nreal*factor,Nreal*factor),dtype=complex)
                    re_Green_tn = np.zeros((tlimit,Nreal*factor,Nreal*factor),dtype=complex)
                    Green_original_tn = np.zeros((tlimit_compare+1,Nreal*factor,Nreal*factor),dtype=complex)
                    real_Green_center_tn = np.zeros((tlimit_compare+1,Nreal),dtype=complex)
                    N = Nreal
                    for i in range(tlimit_compare+1):
                        t = i*0.1
                        Green_snap = get_Green_snapshot_at_center('./{}/Green_sites/'.format(file_name),t,N)
                        
                        if t == 0:
                            overlap = get_ladder_translational_green(Green_snap, int(N/2)-1, path_to_ladder, large_N)

                        Green_tn[i] = get_ladder_translational_green(Green_snap, int(N/2)-1, path_to_ladder, Nreal*factor)
                        Green_original_tn[i] = Green_tn[i]

                    for i in range(tlimit_compare+1):
                        t = 0.1*i
                        N601_U1_t = get_Green_snapshot_at_center('./{}/Green_sites'.format(file_name),t,N=Nreal)
                        real_Green_center_tn[i,:] = N601_U1_t

                    real_Green_center = np.array([real_Green_center_tn[i,int(N/2)-1] for i in range(len(real_Green_center_tn))])
                    dmrg_t = [i*0.1 for i in range(len(real_Green_center_tn))]
                    plt.plot(dmrg_t, real_Green_center,'-',color='black',label='DMRG')

                    lp_order = 5
                    extend_N = tlimit - start_time
                    print(len(real_Green_center), start_time, tlimit, extend_N)
                    Green_lp_real_list = list(burg_interpolation(real_Green_center[0:start_time],lp_order,extend_N))
                    print(len(Green_lp_real_list))
                    # plt.plot([t*0.1 for t in range(len(Green_lp_real_list))],  Green_lp_real_list, c = 'blue', label='LP_'+str(before_time))
                    plt.plot([t*0.1 for t in range(len(Green_lp_real_list))],  Green_lp_real_list, label='LP')

                    for trim_time in [20]:
                        before_time = 0
                        extend_N = tlimit - start_time
                        Green_lp_real_list = list(real_Green_center[0:trim_time])+list(burg_interpolation(real_Green_center[trim_time:start_time],lp_order,extend_N))
                        print(len(Green_lp_real_list))
                        plt.plot([t*0.1 for t in range(len(Green_lp_real_list))],  Green_lp_real_list, label='after LP_'+str(trim_time))
                        print()

                    # # for before_time in [0,5,10,20,40,80]:
                    for before_time in [20]:
                        print(len(real_Green_center))
                        real_Green_center = np.concatenate((np.conj(real_Green_center)[::-1][-before_time::], real_Green_center))
                        print(len(real_Green_center))
                        lp_order = 5
                        extend_N = tlimit - start_time
                        Green_lp_real_list = list(burg_interpolation(real_Green_center[0:start_time+before_time],lp_order,extend_N))[before_time:]
                        print(len(Green_lp_real_list))
                        plt.plot([t*0.1 for t in range(len(Green_lp_real_list))],  Green_lp_real_list, label='before LP_'+str(before_time))
                        print()

                    plt.tick_params(direction='in',labelsize=12)
                    plt.locator_params(nbins=6,axis='y')
                    plt.legend(fontsize=14)
                    plt.xlabel('t',fontsize=16)
                    plt.ylabel("$ReG(x=0,t)$",fontsize=16)
                    plt.ylim([-0.2,0.2])
                    plt.tight_layout()   

plt.savefig('check_LP_doping_{}_U_{}_case_{}.png'.format(doping,U,case))
plt.show()

