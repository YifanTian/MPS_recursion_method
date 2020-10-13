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

def fourier_transform(sample,dag):
    n = len(sample)
    N = n
    Aw = np.zeros(5*N,dtype=np.complex128)
    warray = np.zeros(5*N,dtype=np.complex128)
    for wn in range(5*N):
        for x in range(n):
            w = 2*np.pi*wn/N - 5*np.pi
            warray[wn] = w
            if dag:
                t = (x-int(n/2))*(0.1)                        # right
            else:
                t = (x-int(n/2))*(-0.1)                        # right
            Aw[wn] += 0.1*sample[x]*np.exp(1j*w*t)
    return warray, Aw
    
def fourier_transform_k(sample):
    factor = 1
    n = int(len(sample)/factor)
    nk = 1*int(len(sample)/factor)
    Ak = np.zeros(nk,dtype=np.complex128)    
    
    for k in range(nk):
        fk_vector = [np.exp(-2*np.pi*1j*(x-int(n/2))*(k*1.0)/nk) for x in range(n)]
        Ak[k] += np.dot(np.array(sample), fk_vector)
    return Ak

def fourier_transform_x(sample):
    n = len(sample)
    # nk = 1*len(sample)*2
    nk = 1*len(sample)
    Ak = np.zeros(nk,dtype=np.complex128)
    for k in range(nk):
        for x in range(n):
            m = x-int(n/2)
            # print(type(np.exp(-2*np.pi*1j*m*k/n)))
            # print(sample[x], np.exp(-2*np.pi*1j*m*(k*1.0)/nk), -2*np.pi*1j*m*(k*1.0)/nk)
            Ak[k] += sample[x]*np.exp(2*np.pi*1j*m*(k*1.0)/nk)
            # print(Ak[k])
    return Ak

def fourier_transform_k_matrix(sample):
    n = len(sample)
    nk = 1*len(sample)
    fk_matrix = np.zeros((n,nk),dtype=np.complex128)
    for x in range(n):
        for k in range(nk):
            m = x-int(n/2)
            fk_matrix[x,k] = np.exp(-2*np.pi*1j*m*(k*1.0)/nk)
    return fk_matrix


def complete_evo(occilation):
    occilation = np.array(occilation)
    occilation = np.append(np.conj(np.flip(occilation[1:],0)),occilation)
    return occilation

def get_spectral(green_signal,dag):
    tarray = np.array([i for i in range(len(green_signal))])
    maxT = len(tarray)
    Green_lp_long_real_list = np.real(green_signal)
    Green_lp_long_imag_list = np.imag(green_signal)
    # Green_lp_long_real_list = Green_lp_long_real_list*np.exp(-4*(tarray/maxT)**2)
    # Green_lp_long_imag_list = Green_lp_long_imag_list*np.exp(-4*(tarray/maxT)**2)
    Green_lp_long_real_list = Green_lp_long_real_list*np.exp(-8*(tarray/maxT)**2)
    Green_lp_long_imag_list = Green_lp_long_imag_list*np.exp(-8*(tarray/maxT)**2)
    Green_lp_list = Green_lp_long_real_list + 1j*Green_lp_long_imag_list
    warray, wvalue = fourier_transform(complete_evo(Green_lp_list),dag)
    # print(np.sum(wvalue))
    # print(warray[0],warray[-1])
    wvalue = np.array(wvalue)/(2*np.pi)
    return warray, wvalue


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

        warray, w_value = fourier_transform(real_occilation+1j*img_occilation)

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
    # plt.savefig('N600_U1_real_density.pdf')
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
            # n = float(nums[0]) + 1j*float(nums[1])
            n = float(nums[1]) + 1j*float(nums[2])
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

def complete_evo(occilation):
    occilation = np.array(occilation)
    occilation = np.append(np.conj(np.flip(occilation[1:],0)),occilation)
    return occilation

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

def get_2d_translational_green(green):
    N = len(green)
    Green_temp = np.zeros((N,N),dtype=complex)
    even_green = green
    odd_green = green[::-1]
    even_green_dict = dict()
    odd_green_dict = dict()
    for i in range(N):
        even_green_dict[i-(int(N/2)-1)] = even_green[i]
    for i in range(N):
        odd_green_dict[i-int(N/2)] = odd_green[i]
    for i in range(1,N,2):
        for j in range(N):
            if j - i in even_green_dict:
                Green_temp[i,j] = even_green_dict[j-i]
    for i in range(0,N,2):
        for j in range(N):
            if j - i in odd_green_dict:
                Green_temp[i,j] = odd_green_dict[j-i]
    return Green_temp

def save_dict(d, file_name, warray):
    with open(file_name,'w') as f:
        f.write(' '.join([str(n) for n in warray])+'\n')
        for key in sorted(d):
            f.write(str(key)+'\n')
            f.write(' '.join([str(n) for n in d[key]])+'\n')

def get_recursion(tlimit, re_rstep, green_k_list, weight):
    new_Green_tn = Green_tn*weight
    overlap = new_Green_tn[0]
    overlap_pinv = np.linalg.pinv(overlap,rcond=1e-1)

    new_green_k_list = green_k_list*weight
    coef_matrix = new_Green_tn[re_rstep]
    coef_matrix = np.matmul(overlap_pinv, coef_matrix.T)
    A0k_re = coef_matrix[int(N/2),]     

    re_rnum = int(tlimit/re_rstep)
    re_center_k_val_list = np.zeros(re_rstep*(re_rnum+1)+1,dtype=complex)
    for tau in range(re_rstep):
        re_center_k_val = new_green_k_list[tau]
        re_center_k_val_list[tau] = re_center_k_val
        t = tau
        for rn in range(re_rnum):
            t = t+re_rstep 
            re_target_k_slice = re_center_k_val*center_phase_array
            re_center_k_val = np.dot(A0k_re, re_target_k_slice)
            re_center_k_val_list[t] = re_center_k_val
    return  re_center_k_val_list

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--doping', type=str,
                     help='doping',default='8th')
# parser.add_argument('--start_time', type=int,
#                      help='start_time',default=120)
parser.add_argument('--U', type=int,
                     help='U',default=8)
args = parser.parse_args()

print(args.doping, args.U)

arg_train_time = 100
FT_time = 40*10
evolve_factor = 5
factor = 2

read_dir_name = 'Green_multi_delta_re_matrix_t{}_tlimit{}_factor{}'.format(arg_train_time, arg_train_time*evolve_factor, factor)
save_dir_name = 'long_multi_delta_re_matrix_Akw_t{}_tlimit{}_factor{}'.format(arg_train_time, arg_train_time*evolve_factor, factor)

for dir_name, dag in zip(['{}_results/U{}_dag/'.format(args.doping,args.U)],[True]):
    for ky_0 in [True,False]:


        N_size = 64*factor
        dmrg_k_val_dict = dict()
        recursion_lp_k_val_dict = dict()
        new_recursion_lp_k_val_dict = dict()
        recursion_k_val_dict = dict()
        lp_k_val_dict = dict()
        for total_train_time in [arg_train_time]:

            rstep = int(total_train_time/2)
            tlimit = arg_train_time*evolve_factor
            tlimit_compare = 120

            Nreal = N_size
            N = Nreal-1
            Green_re_tn = np.zeros((tlimit,N,N),dtype=complex)
            Green_continue_re_tn = np.zeros((tlimit,N,N),dtype=complex)
            real_Green_center_tn = np.zeros((tlimit,N),dtype=complex)

            for i in range(tlimit):
                t = i*0.1
                print('read green: ',t)
                if ky_0 == True:
                    continue_Green_re = get_Green_snapshot_at_center('./{}/{}/Green_ky_0'.format(dir_name, read_dir_name),t,Nreal)
                else:
                    continue_Green_re = get_Green_snapshot_at_center('./{}/{}/Green_ky_1'.format(dir_name, read_dir_name),t,Nreal)
                
                continue_Green_re = 0.5*(continue_Green_re[0:-1]+ continue_Green_re[0:-1][::-1])

                Green_temp = np.zeros((N,N),dtype=complex)
                Green_temp[int(N/2),] = continue_Green_re
                Green_continue_re_tn[i] = approximate_matrix(Green_temp,N)

            Gk_array = []
            for i in range(tlimit):
                print('get gk: ', i*0.1)
                center_green_k = fourier_transform_k(Green_continue_re_tn[i,int(N/2),])
                Gk_array.append(center_green_k)
            Gk_array = np.array(Gk_array)


            total_dis_dict = dict()
            total_dis_dict['ReLP1'] = []
            total_dis_dict['ReLP2'] = []
            total_dis_dict['LP'] = []
            total_dis_dict['Re'] = []
            total_dis_dict['real_Re'] = []
            total_dis_dict['Multi_Re'] = []
            k_len = len(Green_continue_re_tn[0])
            k_val_list = []
            for k_index in range(0,k_len,2):
                kidx = int(N/2)-k_index
                k_val = 2*kidx*1.0/k_len
                k_val_list.append(k_val)
                print(k_val)
                
                green_continue_re_k_list = np.array([Gk_array[i,int(N/2)-k_index] for i in range(tlimit)])

                # ========================================================
                import os
                if not os.path.exists('./{}/{}'.format(dir_name, save_dir_name)):
                    os.mkdir('./{}/{}'.format(dir_name, save_dir_name))

                warray, continue_re_wvalue = get_spectral(green_continue_re_k_list[0:FT_time], dag)
                # if k_index % 5 == 0: 
                plt.figure()
                plt.plot([0.1*i for i in range(len(green_continue_re_k_list))], np.real(green_continue_re_k_list),'-', linewidth=2, label='Re real')
                plt.plot([0.1*i for i in range(len(green_continue_re_k_list))], np.imag(green_continue_re_k_list),'-', linewidth=2, label='Re imag')
                plt.axvline(x=0.1*total_train_time,ls='--',c='g')
                plt.legend()
                plt.ylabel("$ReG(k_y=0,t)$",fontsize=16)
                plt.xlabel('t',fontsize=16)
                plt.tick_params(direction='in',labelsize=12)
                plt.tight_layout()  

                if ky_0 == True:
                    plt.savefig('./{}/{}/Green_ky0_k{:.3f}_intial_train_time{}_rstep{}_tlimit{}.pdf'.format(dir_name, save_dir_name, k_val, total_train_time*0.1, rstep*0.1, tlimit))
                else:
                    plt.savefig('./{}/{}/Green_ky1_k{:.3f}_intial_train_time{}_rstep{}_tlimit{}.pdf'.format(dir_name, save_dir_name, k_val, total_train_time*0.1, rstep*0.1, tlimit))

                plt.figure()
                plt.plot(warray, continue_re_wvalue, 'red', label='Multi Recursion')
                plt.legend()
                plt.xlabel('$\omega$',fontsize=16)
                plt.ylabel('$A(k=0,\omega)$',fontsize=16)
                plt.tick_params(direction='in',labelsize=12)
                plt.tight_layout()   
                if ky_0:
                    plt.savefig('./{}/{}/spectral_Akw_ky0_k{:.3f}_trainT_{}_tlimit{}.pdf'.format(dir_name, save_dir_name, k_val, total_train_time*0.1, tlimit*0.1))
                else:
                    plt.savefig('./{}/{}/spectral_Akw_ky1_k{:.3f}_trainT_{}_tlimit{}.pdf'.format(dir_name, save_dir_name, k_val, total_train_time*0.1, tlimit*0.1))

                recursion_k_val_dict[int(k_val*100)/100.0] = continue_re_wvalue

            total_dis_file_name = ''
            if ky_0:
                save_dict(recursion_k_val_dict, "./{}/{}/recursion_ky0_dict.txt".format(dir_name, save_dir_name),warray)
                total_dis_file_name = './{}/{}/ky0_dis_dict.txt'.format(dir_name, save_dir_name)
            else:
                save_dict(recursion_k_val_dict, "./{}/{}/recursion_ky1_dict.txt".format(dir_name,save_dir_name),warray)
                total_dis_file_name = './{}/{}/ky1_dis_dict.txt'.format(dir_name,save_dir_name)

            with open(total_dis_file_name,'w') as f:
                for key in sorted(total_dis_dict):
                    f.write(str(key)+' '+str(np.mean(total_dis_dict[key]))+'\n')
                    f.write(str(key)+' '+str(total_dis_dict[key])+'\n')
