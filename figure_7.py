import numpy as np
import matplotlib.pyplot as plt
import spectrum

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
    for ii in range(M,P):    
        y[ii] = -sum(a[1:] * y[(ii-1):(ii-N-1):-1] )
        print(y[ii])
    return y

def fourier_transform(sample,dag):
    n = len(sample)
    # N = 2*n
    N = n
    Aw = np.zeros(4*N,dtype=np.complex128)
    warray = np.zeros(4*N,dtype=np.complex128)
    for wn in range(4*N):
        for x in range(n):
            w = 2*np.pi*wn/N - 4*np.pi
            warray[wn] = w
            if dag:
                t = (x-int(n/2))*(0.1)                        # right
            else:
                t = (x-int(n/2))*(-0.1)                        # right
            Aw[wn] += 0.1*sample[x]*np.exp(1j*w*t)
    return warray, Aw

def fourier_transform_k(sample):
    n = len(sample)
    # nk = 1*len(sample)*2
    nk = 1*len(sample)
    Ak = np.zeros(nk,dtype=np.complex128)
    for k in range(nk):
        for x in range(n):
            m = x-int(n/2)
            # print(type(np.exp(-2*np.pi*1j*m*k/n)))
            # print(sample[x], np.exp(-2*np.pi*1j*m*(k*1.0)/nk), -2*np.pi*1j*m*(k*1.0)/nk)
            Ak[k] += sample[x]*np.exp(-2*np.pi*1j*m*(k*1.0)/nk)
            # print(Ak[k])
    return Ak

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

def get_spectral(green_signal,dag):
    tarray = np.array([i for i in range(len(green_signal))])
    maxT = len(tarray)
    Green_lp_long_real_list = np.real(green_signal)
    Green_lp_long_imag_list = np.imag(green_signal)

    Green_lp_long_real_list = Green_lp_long_real_list*np.exp(-4*(tarray/maxT)**2)
    Green_lp_long_imag_list = Green_lp_long_imag_list*np.exp(-4*(tarray/maxT)**2)

    # Green_lp_long_real_list = Green_lp_long_real_list*np.exp(-8*(tarray/maxT)**2)
    # Green_lp_long_imag_list = Green_lp_long_imag_list*np.exp(-8*(tarray/maxT)**2)

    Green_lp_list = Green_lp_long_real_list + 1j*Green_lp_long_imag_list
    warray, wvalue = fourier_transform(complete_evo(Green_lp_list),dag)
    # print(np.sum(wvalue))
    # print(warray[0],warray[-1])
    wvalue = np.array(wvalue)/(2*np.pi)
    # wvalue = np.array(wvalue)/1000.0
    # print(np.sum(wvalue), warray[0] - warray[-1])
    return warray, wvalue

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

factor = 1
# clist = ['red','green','blue','black']
style_list = ['--','-.','-',':']
# get the green at one k and get the spectral from green in real
# k_index = 16
# dir_name_list = ['half_results/U8/','4th_results/U8/','8th_results/U8/']
dir_name_list = ['4th_results/U8/']
for dir_name in dir_name_list:
    # for k_val in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    # for k_val in [0.3,0.4]:
    for k_val in [0.3]:
        fig, axs = plt.subplots(2, 1,  figsize=(8,10))
        # for k_val in [0.5,0.6,0.7,0.8,0.9,1.0]:
        # for k_val in [0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5]:
        # for k_val in [0.5]:
        # k_val = 0.3
        # arg_train_time = 150
        Nreal = 64*factor
        N = Nreal-1
        # dir_name = '4th_results/U8/'
        if dir_name[-4:-1] == 'dag':
            dag = True 
        else:
            dag = False
        print(dir_name[-4:-1])
        ky_0 = True
        # ky_0 = False
        k_index = int(int(N/2) - (k_val*N/2))
        kidx = int(N/2)-k_index
        k_val = 2*kidx*1.0/N
        print('k_val: ',k_val)
        print(k_index)

        def get_akw_from_dir(read_dir_name, k_val, train_time, factor):
            tlimit = train_time*factor
            re_read_dir_name = 'Green_multi_re_t{}'.format(train_time)
            Green_continue_re_tn = np.zeros((tlimit,N,N),dtype=complex)
            for i in range(tlimit):
                t = i*0.1
                if ky_0 == True:
                    continue_Green_re = get_Green_snapshot_at_center('./{}/{}/Green_ky_0'.format(dir_name, re_read_dir_name),t,Nreal)
                else:
                    continue_Green_re = get_Green_snapshot_at_center('./{}/{}/Green_ky_1'.format(dir_name, re_read_dir_name),t,Nreal)
                
                continue_Green_re = 0.5*(continue_Green_re[0:-1]+ continue_Green_re[0:-1][::-1])

                Green_temp = np.zeros((N,N),dtype=complex)
                Green_temp[int(N/2),] = continue_Green_re
                Green_continue_re_tn[i] = approximate_matrix(Green_temp,N)

            green_continue_re_k_list = []
            for i in range(tlimit):
                center_green_k = fourier_transform_k(Green_continue_re_tn[i,int(N/2),])
                green_continue_re_k_list.append(center_green_k[int(N/2)-k_index])
                print(i*0.1)
            green_continue_re_k_list = np.array(green_continue_re_k_list) 

            # plt.plot([0.1*i for i in range(len(green_continue_re_k_list))], np.real(green_continue_re_k_list),'-', c='red', linewidth=2, label='Re')
            # plt.show()

            warray, continue_re_wvalue = get_spectral(green_continue_re_k_list, dag)
            # plt.plot(warray, continue_re_wvalue, 'red', label='Recursion')
            # plt.xlim([-5,5])
            # plt.show()
            return green_continue_re_k_list, warray, continue_re_wvalue

        def read_dmrg(read_dir_name, k_val, train_time):
            factor = 1
            tlimit = train_time*factor
            
            Green_tn = np.zeros((tlimit,N,N),dtype=complex)
            for i in range(train_time):
                t = i*0.1
                if ky_0 == True:
                    Green_TDVP = get_Green_snapshot_at_center('./{}/Green_sites/Green_ky_0'.format(read_dir_name),t,Nreal)
                else:
                    Green_TDVP = get_Green_snapshot_at_center('./{}/Green_sites/Green_ky_1'.format(read_dir_name),t,Nreal)
                
                Green_TDVP = 0.5*(Green_TDVP[0:-1]+ Green_TDVP[0:-1][::-1])

                Green_temp = np.zeros((N,N),dtype=complex)
                Green_temp[int(N/2),] = Green_TDVP
                Green_tn[i] = approximate_matrix(Green_temp,N)

            center_green_k_list = []
            for i in range(train_time):
                center_green_k = fourier_transform_k(Green_tn[i,int(N/2),])
                center_green_k_list.append(center_green_k[int(N/2)-k_index])
            center_green_k_list = np.array(center_green_k_list)                 # the recursion k val
 
            warray, dmrg_wvalue = get_spectral(center_green_k_list, dag)
            # plt.plot(warray, lp_wvalue, 'blue', label='LP for t > {}'.format(int(train_time*0.1)))
            # plt.title('Re {:.2f} | LP{:.2f}'.format(Re_w_distance, LP_w_distance))
            # plt.legend()
            # plt.show()
            return center_green_k_list, warray, dmrg_wvalue

        # get the green at one k and get the spectral from green in real using linear prediction
        lp_read_dir_name = '../ladder_system_test/{}'.format(dir_name)
        def get_lp_akw_from_dir(read_dir_name, k_val, train_time, factor):
            tlimit = train_time*factor
            
            Green_tn = np.zeros((tlimit,N,N),dtype=complex)
            for i in range(train_time):
                t = i*0.1
                if ky_0 == True:
                    Green_TDVP = get_Green_snapshot_at_center('./{}/Green_sites/Green_ky_0'.format(read_dir_name),t,Nreal)
                else:
                    Green_TDVP = get_Green_snapshot_at_center('./{}/Green_sites/Green_ky_1'.format(read_dir_name),t,Nreal)
                
                Green_TDVP = 0.5*(Green_TDVP[0:-1]+ Green_TDVP[0:-1][::-1])

                Green_temp = np.zeros((N,N),dtype=complex)
                Green_temp[int(N/2),] = Green_TDVP
                Green_tn[i] = approximate_matrix(Green_temp,N)

            center_green_k_list = []
            for i in range(train_time):
                center_green_k = fourier_transform_k(Green_tn[i,int(N/2),])
                center_green_k_list.append(center_green_k[int(N/2)-k_index])
            center_green_k_list = np.array(center_green_k_list)                 # the recursion k val

            extend_N = tlimit - train_time
            lp_order = 5
            Green_lp_real_list = list(burg_interpolation(center_green_k_list[0:train_time],lp_order,extend_N))

            warray, lp_wvalue = get_spectral(Green_lp_real_list, dag)
            return Green_lp_real_list, warray, lp_wvalue

        clist = ['black','red']
        # plt.figure()
        cidx = 0
        idx = 0
        for train_time,factor in zip([60,120],[4,2]):
            for k_val in [k_val]: 
                green_continue_re_k_list, warray, continue_re_wvalue = get_akw_from_dir('', k_val, train_time, factor)
                axs[0].plot([0.1*i for i in range(len(green_continue_re_k_list))], np.real(green_continue_re_k_list), style_list[idx], label='Recursion t > {}'.format(int(train_time*0.1)))
                idx += 1
            cidx += 1

        clist = ['blue','green']
        total_train_time = 240
        cidx = 0
        for train_time,factor in zip([60,120],[4,2]):
            for k_val in [k_val]:
                Green_lp_real_list, warray, lp_wvalue = get_lp_akw_from_dir(lp_read_dir_name, k_val, train_time, factor)
                # plt.plot([0.1*i for i in range(len(Green_lp_real_list))], np.real(Green_lp_real_list), label='LP for t > {} factor {}'.format(int(train_time*0.1), factor))
                if style_list[idx] == ':':
                    axs[0].plot([0.1*i for i in range(len(Green_lp_real_list))], np.real(Green_lp_real_list), style_list[idx], linewidth=2, label='LP t > {}'.format(int(train_time*0.1)), markersize=1)
                else:
                    axs[0].plot([0.1*i for i in range(len(Green_lp_real_list))], np.real(Green_lp_real_list), style_list[idx], label='LP t > {}'.format(int(train_time*0.1)), markersize=1)
                idx += 1
            cidx += 1

        clist = ['black','red']
        # plt.figure()
        total_train_time = 240
        cidx = 0
        idx = 0
        for train_time,factor in zip([60,120],[4,2]):
            for k_val in [k_val]:
                green_continue_re_k_list, warray, continue_re_wvalue = get_akw_from_dir('', k_val, train_time, factor)
                axs[1].plot(warray, continue_re_wvalue, style_list[idx], label='Recursion t > {}'.format(int(train_time*0.1)))
                idx += 1
            cidx += 1
        # plt.legend()
        plt.xlim([-5,5])
        plt.xlabel('$\omega$',fontsize=16)
        plt.ylabel('$A(k_y=0,\omega)$',fontsize=16)
        
        clist = ['blue','green']
        cidx = 0
        for train_time,factor in zip([60,120],[4,2]):
            for k_val in [k_val]:   
                Green_lp_real_list, warray, lp_wvalue = get_lp_akw_from_dir(lp_read_dir_name, k_val, train_time, factor)
                # plt.plot(warray, lp_wvalue, label='LP for t > {} factor {}'.format(int(train_time*0.1), factor))
                if style_list[idx] == ':':
                    axs[1].plot(warray, lp_wvalue, style_list[idx], linewidth = 2, label='LP t > {}'.format(int(train_time*0.1)), markersize=1)
                else:
                    axs[1].plot(warray, lp_wvalue, style_list[idx], label='LP t > {}'.format(int(train_time*0.1)), markersize=1)
                idx += 1
            cidx += 1

        dmrg_green_k_list, warray, dmrg_wvalue = read_dmrg(lp_read_dir_name, k_val, 150)

        axs[0].plot([0.1*i for i in range(len(dmrg_green_k_list))], dmrg_green_k_list, linestyle=(0, (3, 1, 1, 1)), c='black', label = 'DMRG')
        
        axs[0].axvline(x=6, c='b', ls='--',linewidth=1.0)
        axs[0].axvline(x=12, c='b', ls='--',linewidth=1.0)
        axs[1].plot(warray, dmrg_wvalue, linestyle=(0, (3, 1, 1, 1)), c='black', label = 'DMRG')

        axs[0].text(7.5,0.05,'(a)',fontsize=20)
        axs[0].set_xlabel('t',fontsize=18)
        axs[0].set_xticks(np.arange(4, 24.1, 4))
        axs[0].set_ylabel("$ReG(k_y=0,t)$",fontsize=18)
        axs[0].set_xlim([4,24])
        axs[0].set_ylim([-0.15,0.1])
        axs[0].legend(fontsize=14)
        axs[0].tick_params(direction='in',labelsize=15)
        axs[1].tick_params(direction='in',labelsize=15)

        # axs[0].set_xlim([-5,5])
        axs[1].text(1.0,0.4,'(b)',fontsize=20)
        axs[1].set_xlim([-5,2])
        axs[1].legend(fontsize=14)
        axs[1].set_xlabel('$\omega$',fontsize=18)
        axs[1].set_ylabel('$A(k_y=0,\omega)$',fontsize=18)
        # plt.title('$k={:.2f}\pi$'.format(k_val))
        plt.tight_layout()
        plt.savefig('./figures/re_lp_{}_spectral_k{:.2f}_ky0_train_time{}_compare_new_style1.pdf'.format('_'.join(dir_name.split('/')), k_val, total_train_time))
        plt.show()

        # print(green_continue_re_k_list, Green_lp_real_list)


