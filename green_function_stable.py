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

# def fourier_transform(sample):
#     n = len(sample)
#     N = 2*n
#     Aw = np.zeros(N,dtype=np.complex128)
#     warray = np.zeros(N,dtype=np.complex128)
#     for wn in range(N):
#         for x in range(n):
#             w = 2*np.pi*wn/N
#             warray[wn] = w
#             t = (x-int(n/2))*0.1                        # right
#             Aw[wn] += 0.1*sample[x]*np.exp(1j*w*t)
#     return warray, Aw

def fourier_transform(sample):
    n = len(sample)
    N = 2*n
    Aw = np.zeros(2*N,dtype=np.complex128)
    warray = np.zeros(2*N,dtype=np.complex128)
    # warray = np.zeros(N,dtype=np.complex128)
    # for wn in range(N):
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
    # print(y,data[0:M])
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
                # img_profile = 0.5*(img_profile[0:-1]+ img_profile[0:-1][::-1])
                # real_profile = 0.5*(real_profile[0:-1]+ real_profile[0:-1][::-1])

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
        
        # plt.plot(real_occilation,label='real')
        # plt.plot(img_occilation,label='imag')
        # np.savetxt('A_i_occilation.txt', (real_occilation,img_occilation), delimiter=',')   # X is an array
        # plt.legend()
        # plt.show()

        warray, w_value = fourier_transform(real_occilation+1j*img_occilation)
        # plt.plot(warray,w_value)
        # plt.show()
        # warray, w_value = fourier_transform(real_occilation)

        # w_value = DFT(real_occilation+1j*img_occilation)
        # w_value = np.fft.fft(real_occilation+1j*img_occilation)
        # warray = [val/2 for val in warray]
        print(max(warray)-min(warray))
        print(sum(w_value)/len(warray)*(max(warray)-min(warray)))
        print(np.max(w_value))
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
    
    # plt.plot(center_signal)
    # plt.plot(new_mm[int(len(mm)/2),:])
    # plt.show()
    return new_mm

def approximate_matrix_odd_even(mm, N):
    # center_signal = mm[int(len(mm)/2),:]
    # center_signal = mm[int(len(mm)/2)-1,:]
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
    # if mode == 'analytic':
    #     file_name = './N{}/Green_center/Green_t{:.3f}.txt'.format(N,t)
    # else:
    #     file_name = './N{}/Green_DMRG/Green_t{:.3f}.txt'.format(N,t)
    # if mode == 'smooth':
    #     file_name = './N{}_smooth/Green_center/Green_t{:.3f}.txt'.format(N,t)
    # else:
    #     file_name = './N{}/Green_center/Green_t{:.3f}.txt'.format(N,t)
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
    # plt.plot(data, linewidth=0.8)
    # plt.show()
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
        # relative_idx = (relative_idx[0],relative_idx[1])
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
# parser.add_argument('--start_time', type=int,
#                      help='start_time',default=120)
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
    # print(n, ix, iy)
    if iy == 0:
        leg0_sites.append(n)
    else:
        leg1_sites.append(n)
    path_to_ladder[n] = (ix,iy)
    if n%2 == 0:
        iy = 1-iy
    else:
        ix+=1
    
import gc
evolve_facotr = 4
# file_name = '8th_results/U8_dag/'
# for doping in ['4th']:
# for doping in ['half','4th','8th']:
for doping in ['half']:
#     for case in ['U4','U4_dag']:
# new_dir_name = './4th_results'
# for files in ['../ladder_system_test/{}_results'.format(args.doping)]:
    files = '../ladder_system_test/{}_results'.format(doping)
    new_dir_name = files.split('/')[-1]
    # for case in ['U{}'.format(args.U),'U{}_dag'.format(args.U)]:
    # for U in [1,2,4,8]:
    for U in [args.U]:
        # for case in ['U{}'.format(args.U),'U{}_dag'.format(args.U)]:
        # for case in ['U{}'.format(U),'U{}_dag'.format(U)]:
        for case in ['U{}_dag'.format(U)]:
            file_name = files+'/'+case+'/'
            muti_num = 6
            interval_step = 10
            start_time = args.start_time
            muti_num_grid = [10]
            interval_step_grid = [5]
            step_size = interval_step_grid[0]
            res_grid = np.zeros([10, len(interval_step_grid)])
            dis_grid = np.zeros([10, len(interval_step_grid)])
            res_step_list = []
            dis_step_list = []
            m_idx = 0
            # plt.figure()
            # for start_time in [30,40,50,60,70,80]:
            # for start_time in [60]:
            for m_idx, muti_num in enumerate(muti_num_grid):
            # for alpha in [0.1,1.0,10.0,50.0,60.0,70.0,80.0,100.0]:
            # for alpha in [50.0,60.0]:
                # m_idx = 0
                # muti_num = 10
                for i_idx, interval_step in enumerate(interval_step_grid):
                    # muti_num = m_idx
                    recursion_step = start_time - (muti_num-1)*interval_step
                    if recursion_step <= 0:
                        res_grid[m_idx, i_idx] = res_grid[0, 0]
                        dis_grid[m_idx, i_idx] = dis_grid[0, 0]
                        continue

                    tlimit_compare = 150
                    tlimit = 500
                    extend_N = 0
                    Nreal = 128
                    factor = 2
                    large_GN = factor*Nreal
                    large_N = factor*Nreal
                    
                    overlap_Green = np.zeros((tlimit,large_N,large_N),dtype=complex)
                    Green_tn = np.zeros((tlimit,Nreal*factor,Nreal*factor),dtype=complex)
                    re_Green_tn = np.zeros((tlimit,Nreal*factor,Nreal*factor),dtype=complex)
                    re_new_Green_tn = np.zeros((tlimit,Nreal*factor,Nreal*factor),dtype=complex)
                    re_delta_new_Green_tn = np.zeros((tlimit,Nreal*factor,Nreal*factor),dtype=complex)
                    Green_original_tn = np.zeros((tlimit_compare+1,Nreal*factor,Nreal*factor),dtype=complex)
                    real_Green_center_tn = np.zeros((tlimit_compare+1,Nreal),dtype=complex)
                    N = Nreal
                    for i in range(tlimit_compare+1):
                        t = i*0.1
                        Green_snap = get_Green_snapshot_at_center('./{}/Green_sites/'.format(file_name),t,N)
                                                
                        if t == 0:
                            overlap = get_ladder_translational_green(Green_snap, int(N/2)-1, path_to_ladder, large_N)

                        print('i: ',i)
                        Green_tn[i] = get_ladder_translational_green(Green_snap, int(N/2)-1, path_to_ladder, Nreal*factor)
                        Green_original_tn[i] = Green_tn[i]

                    print('overlap.shape: ',overlap.shape)
                    overlap_pinv = np.linalg.pinv(overlap,rcond=1e-1)

                    N_total = Nreal*factor
                    Ci = dict()
                    weight_list = [1.0 for _ in range(muti_num)]
                    residual = np.zeros((tlimit_compare+1,Nreal*factor,Nreal*factor),dtype=complex)
                    new_residual = np.zeros((tlimit_compare+1,Nreal*factor,Nreal*factor),dtype=complex)
                    t_len = len(weight_list)
                    for i in range(recursion_step+((t_len)-1)*interval_step):
                        re_Green_tn[i,:,:] = Green_tn[i,:,:]
                        re_new_Green_tn[i,:,:] = Green_tn[i,:,:]
                        re_delta_new_Green_tn[i,:,:] = Green_tn[i,:,:]
                    for i in range(tlimit_compare):
                        residual[i,:] = Green_original_tn[i,:]
                    t_list = []
                    residual_dict = dict()
                    t1_list = []

                    residual_list_dict = dict()
                    green_list_dict = dict()
                    cum_green_list= [0.0 for t in range(tlimit_compare)]

                    for j in range(t_len):
                        t1 = recursion_step+j*interval_step
                        print('t1: ',t1)
                        Ci[t1] = np.matmul(residual[t1], overlap_pinv)        # this should be full
                        for t in range(t1, tlimit_compare):
                            new_residual[t] = residual[t] - np.matmul(Ci[t1], Green_original_tn[t-t1])        # this should have weight
                            residual_list_dict[t] = new_residual[t]
                            cum_green_list[t] += np.matmul(Ci[t1], Green_original_tn[t-t1])
                        t1_list.append(t1*0.1)
                        residual_dict[t1] = new_residual[t1]
                        residual= new_residual.copy()

                    green_list = dict()
                    green_cum_list = dict()
                    for tm in range(recursion_step,recursion_step+t_len):
                        green_list[tm] = dict()
                        green_cum_list[tm] = dict()
                        for t in range(recursion_step+t_len):
                            green_cum_list[tm][t] = re_Green_tn[t].copy()
                            green_list[tm][t] = re_Green_tn[t].copy()

                    nl = int(10/0.5)
                    Ci_LP_matrix = np.zeros([large_N*nl,large_N*nl],dtype=complex)
                    for idx, t in enumerate(sorted(Ci)):
                        Ci_LP_matrix[0:large_N,(nl-len(Ci)+idx)*large_N:(nl-len(Ci)+idx+1)*large_N] = Ci[t]

                    for idx in range(nl):
                        if idx+1 < nl:
                            Ci_LP_matrix[(idx+1)*large_N:(idx+2)*large_N,idx*large_N:(idx+1)*large_N] = np.diag([1.0 for _ in range(large_N)])
                    
                    print('Ci_LP_matrix: ',Ci_LP_matrix.shape)
                    compute_before = False
                    if not compute_before:
                        s, w = np.linalg.eig(Ci_LP_matrix)
                        np.save('./recursion_matrix/doping_{}_case_{}_eigvals.npy'.format(doping, case), s)
                        np.save('./recursion_matrix/doping_{}_case_{}_eigvectors.npy'.format(doping, case), w)
                    else:
                        print('start loading')
                        s = np.load('./recursion_matrix/doping_{}_case_{}_eigvals.npy'.format(doping, case))
                        w = np.load('./recursion_matrix/doping_{}_case_{}_eigvectors.npy'.format(doping, case))
                        print('after loading')
                    new_s = s
                    new_s = []
                    delta_s = []
                    print('eig')

                    count = 0
                    for i in range(len(s)):
                        if np.abs(s[i]) > 1.0:
                            print(s[i])
                            count += 1
                            new_s.append(1.0/np.conj(s[i]))
                            delta_s.append( s[i]/(np.abs(s[i])*1.000001)-s[i] )
                        else:
                            new_s.append(s[i])
                            delta_s.append(0.0)

                    circle_x_list = []
                    circle_y_list = []
                    import math
                    for i in range(100):
                        circle_x_list.append(math.cos((i/100.0)*2*np.pi)) 
                    
                    w_inv = np.linalg.pinv(w,rcond=1e-5)
                    delta_Ci_LP_matrix = np.matmul(np.matmul(w, np.diag(delta_s)),w_inv)

                    del w
                    del w_inv

                    new_delta_Ci_LP_matrix = Ci_LP_matrix + delta_Ci_LP_matrix
                    del Ci_LP_matrix
                    gc.collect()
                    
                    new_delta_green_array = dict()
                    delta_green_array = dict()
                    for t in range(start_time):
                        new_delta_green_array[t] = Green_tn[t,:,:]
                        delta_green_array[t] = Green_tn[t,:,:]
                    step_t = 5
                    for step_t in range(interval_step_grid[0]):
                        print('step_t: ',step_t)
                        Ci_LP_vector = np.zeros([large_N*nl,large_N],dtype=complex)
                        for idx, t in enumerate([start_time-(i+1)*step_size+step_t for i in range(nl)]):
                            Ci_LP_vector[idx*large_N:(idx+1)*large_N,:] = re_Green_tn[t]

                        new_Ci_LP_vector = np.empty_like(Ci_LP_vector)
                        new_Ci_LP_vector[:] = Ci_LP_vector
                        
                        new_delta_Ci_LP_vector = np.empty_like(new_Ci_LP_vector)
                        new_delta_Ci_LP_vector[:] = new_Ci_LP_vector
                        
                        delta_Ci_LP_vector = np.empty_like(new_delta_Ci_LP_vector)
                        delta_Ci_LP_vector[:] = new_delta_Ci_LP_vector

                        green_center_list = []
                        new_green_center_list = []
                        new_delta_green_center_list = []
                        
                        rescaled_green_list = []
                        LP_time_list = []
                        idx_t = 0
                        for t in range(start_time,tlimit,step_size):
                            print('t: ',t)

                            delta_Ci_LP_vector = np.matmul(new_delta_Ci_LP_matrix, delta_Ci_LP_vector)
                            delta_green_at_t = delta_Ci_LP_vector[0:large_N,:]

                            LP_time_list.append(start_time*0.1+1.0*idx_t+step_t*0.1)
                            delta_green_array[start_time+step_size*idx_t+step_t*1] = delta_green_at_t
                            rescaled_green_list.append(delta_green_at_t)
                            idx_t += 1
                    
                    del new_delta_Ci_LP_matrix
                    gc.collect()

                    norm_scale_list = []
                    for i in range(len(rescaled_green_list)):
                        norm_scale_list.append(np.sum(np.abs(rescaled_green_list[i]))/np.sum(np.abs(rescaled_green_list[-1])))

                    print('LP_time_list: ',LP_time_list)
                    plt.plot(LP_time_list, norm_scale_list,'--',label='norm')

                    import os
                    green_save_dir = '{}/{}/Green_multi_delta_re_matrix_t{}_tlimit{}_factor{}'.format(new_dir_name, case, start_time, tlimit, factor)
                    if not os.path.isdir(green_save_dir):
                        os.mkdir(green_save_dir)

                    for t in range(tlimit):
                        with open('{}/Green_t{:.3f}.txt'.format(green_save_dir, t*0.1),'w') as f:
                            for i in range(large_GN):
                                f.write(str(np.real(delta_green_array[t][int(large_GN/2)-1,i])) + ' ' + str(np.imag(delta_green_array[t][int(large_GN/2)-1,i])) + '\n' )

                    delta_time_list = [t for t in sorted(delta_green_array.keys())]
                    plt.plot([t*0.1 for t in delta_time_list], [delta_green_array[t][int(large_GN/2)-1,int(large_GN/2)-1] for t in delta_time_list],'-',label='delta self matrix')

                    re_Green_tn = np.zeros((tlimit,Nreal*factor,Nreal*factor),dtype=complex)
                    for i in range(recursion_step+((t_len)-1)*interval_step):
                        re_Green_tn[i,:,:] = Green_tn[i,:,:]

                    new_s = []
                    for i in range(len(s)):
                        if np.abs(s[i]) > 1.0:
                            new_s.append(1.0/s[i])
                        else:
                            new_s.append(s[i])

                    for t in range(recursion_step+((t_len)-1)*interval_step,tlimit):
                        for j in range(t_len):
                            tm = recursion_step+j*interval_step
                            if t-tm >= 0:
                                re_Green_tn[t] += np.matmul(Ci[tm],re_Green_tn[t-tm])

                    print('Ci: ',Ci.keys())
                    interval_step_time_list = []
                    interval_Green_list = []
                    interval_Green_tn_dict = dict()

                    for t in range(0,start_time,interval_step):
                        interval_Green_tn_dict[t] = re_Green_tn[t]
                        
                    for t in range(start_time,start_time+interval_step*10,interval_step):
                        interval_step_time_list.append(t*0.1)
                        for j in range(t_len):
                            tm = recursion_step+j*interval_step
                            if j == 0:
                                interval_Green_tn_dict[t] = 0
                            interval_Green_tn_dict[t] += np.matmul(Ci[tm],interval_Green_tn_dict[t-tm])
                            
                    for t in range(start_time,start_time+interval_step*10,interval_step):
                        interval_Green_list.append(interval_Green_tn_dict[t][int(large_GN/2)-1,int(large_GN/2)-1])

                    re_t = [i*0.1*1.0 for i in range(tlimit)]
                    center_evolve = np.array([re_Green_tn[i,int(large_GN/2)-1,int(large_GN/2)-1]*1.0 for i in range(tlimit)])
                    plt.plot(re_t, center_evolve,'-',label='Re')
                    plt.xlabel('t',fontsize=16)
                    plt.ylabel("$ReG(x=0,t)$",fontsize=16)
                    print(interval_step_time_list)
                    print(interval_Green_list)
                    print(len(interval_Green_list))
                    norm_scale_list = []
                    for i in range(tlimit):
                        norm_scale_list.append(np.sum(np.abs(re_Green_tn[i]))/np.sum(np.abs(re_Green_tn[-1])))

                    for t in interval_Green_tn_dict:
                        u, s, vh = np.linalg.svd(interval_Green_tn_dict[t])
                        print(s[0:10])
                    
                    plt.legend()
                    # plt.savefig('./instability/8th_U{}_large_re_matrix_multi_num{}_space{}_evolution_compare.pdf'.format(args.U, muti_num,interval_step))
                    plt.show()

