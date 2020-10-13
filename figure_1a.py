import numpy as np 
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy.linalg import inv
import matplotlib.animation as animation

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
        
        # plt.plot(real_occilation,label='real')
        # plt.plot(img_occilation,label='imag')
        # np.savetxt('A_i_occilation.txt', (real_occilation,img_occilation), delimiter=',')   # X is an array
        # plt.legend()
        # plt.show()

        warray, w_value = fourier_transform(real_occilation+1j*img_occilation)
        # plt.plot(warray,w_value)
        # plt.show()
        # raise SystemExit(0)
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
    size = len(mm)
    assert len(mm) == len(mm[0,:])
    new_mm = mm.copy()
    val_dict = dict()
    even_val_dict = dict()
    odd_val_dict = dict()

    for i in range(int(size/2)+1):
        even_val_dict[abs(i-int(len(mm)/2))] = mm[int(len(mm)/2),:][i]
    for i in range(int(size/2)+1):
        odd_val_dict[abs(i-int(len(mm)/2)+1)] = mm[int(len(mm)/2)-1,:][i]

    for i in range(N):
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


# for files in ['4th_results','8th_results','half_results']:
for files in ['8th_results']:
    for case in ['U8']:
        file_name = files+'/'+case+'/'
        muti_num = 6
        interval_step = 5
        start_time = 60
        muti_num_grid = [1]
        interval_step_grid = [5]
        res_grid = np.zeros([151, len(interval_step_grid)])
        dis_grid = np.zeros([151, len(interval_step_grid)])
        res_step_list = []
        dis_step_list = []
        m_idx = 0
        for m_idx, muti_num in enumerate(muti_num_grid):
            for i_idx, interval_step in enumerate(interval_step_grid):
                print()
                recursion_step = start_time - (muti_num-1)*interval_step
                if recursion_step <= 0:
                    res_grid[m_idx, i_idx] = res_grid[0, 0]
                    dis_grid[m_idx, i_idx] = dis_grid[0, 0]
                    continue

                # tlimit_compare = 150
                tlimit_compare = 150
                tlimit = 150*2
                extend_N = 0
                Nreal = 128
                Nshort = 128
                Nlarge = Nshort-1+extend_N*2
                Green_tn = np.zeros((tlimit,Nreal,Nreal),dtype=complex)
                re_Green_tn = np.zeros((tlimit,Nreal,Nreal),dtype=complex)
                Green_original_tn = np.zeros((tlimit_compare+1,Nreal,Nreal),dtype=complex)
                real_Green_center_tn = np.zeros((tlimit_compare+1,Nreal),dtype=complex)
                N = Nreal
                for i in range(tlimit_compare+1):
                    t = i*0.1
                    Green_snap = get_Green_snapshot_at_center('./{}/Green_sites/'.format(file_name),t,N)
                    Green_tn[i] = get_2d_translational_green(Green_snap)
                    Green_original_tn[i] = Green_tn[i]

                N = Nreal
                overlap = Green_tn[0]
                overlap_pinv = np.linalg.pinv(overlap,rcond=1e-1)

                Ci = dict()
                weight_list = [1.0 for _ in range(muti_num)]
                residual = np.zeros((tlimit_compare+1,Nreal,Nreal),dtype=complex)
                new_residual = np.zeros((tlimit_compare+1,Nreal,Nreal),dtype=complex)
                t_len = len(weight_list)
                for i in range(recursion_step+((t_len)-1)*interval_step):
                    re_Green_tn[i,:,:] = Green_tn[i,:,:]
                for i in range(tlimit_compare):
                    residual[i,:] = Green_original_tn[i,:]
                t_list = []
                residual_dict = dict()
                t1_list = []

                residual_list_dict = dict()
                green_list_dict = dict()
                cum_green_list= [0.0 for t in range(tlimit_compare)]
                plt.plot([t*0.1 for t in range(tlimit_compare)], [Green_original_tn[t][int(N/2)-1,int(N/2)-1] for t in range(tlimit_compare)],'black',label='DMRG')        
                
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

            plt.plot([t*0.1 for t in range(t1, tlimit_compare)], [cum_green_list[t][int(N/2)-1,int(N/2)-1] for t in range(t1, tlimit_compare)],'red', label='One-step Recursion')
            plt.legend()
            plt.xlabel('t',fontsize=16)
            plt.ylabel("$ReG(x=0,t)$",fontsize=16)
            plt.tick_params(direction='in',labelsize=14)
            plt.locator_params(axis='y', nbins=6)

            plt.ylim([-0.2,0.6])
            plt.xlim([1,15])
            a = plt.axes([0.25, .5, .3, .3])
            
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                left=False,         # ticks along the top edge are off
                labelbottom=True) # labels along the bottom edge are off
            plt.plot([t*0.1 for t in range(t1, tlimit_compare)], [-1.0*new_residual[t][int(N/2)-1,int(N/2)-1] for t in range(t1, tlimit_compare)], label='Error')

            
            time_points = [t*0.1 for t in range(t1, tlimit_compare)]
            residual_points = [new_residual[t][int(N/2)-1,int(N/2)-1] for t in range(t1, tlimit_compare)]

        plt.legend()
        plt.tight_layout()
        plt.savefig('./figure_data/single_recursion_{}_single_discontinuity_1.pdf'.format('_'.join(file_name.split('/'))))
        plt.show()

