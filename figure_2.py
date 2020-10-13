import numpy as np 
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy.linalg import inv
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
            # plt.plot(warray, w_value)

            # x = [x for x in range(7)]
            # kvals = [2*np.cos(k*2*np.pi/21.0) for k in x]
            # x = [x for x in [2,3,4]]
            # kvals = [-2*(np.cos(k*2*np.pi/7.0)) for k in x]
            # print(kvals)
            # ys = [1 for i in kvals]
            # H_mat = get_H_mat(N)
            # w, v = LA.eig(H_mat)
            # for kval in w:
            #     if kval > 0:
            #         plt.axvline(x=kval,c='r')

            # for i in range(N+1):
            #     if i % 2 == 1:
            #         plt.axvline(-2*np.cos(np.pi*i/(401+1)), c='r' )
        idx+=1
    plt.legend()
    # plt.savefig('N600_U1_real_density.pdf')
    plt.show()  

    # file_root = file_name.split('.')[0]
    # file_name = '{}_warray.txt'.format(file_root)
    # file_root = file_name.split('.')[0]

    # file_name = dir_name+'warray.txt'
    # save_file(file_name,warray,w_value)

    wdensity = np.stack([np.array(warray),np.array(w_value)])
    np.save(dir_name+'warray', wdensity)
    # np.save(dir_name+'warray_{}'.format(time_limit), wdensity)
    # if show:
    #     plt.savefig('N401_U1_r100_30step_density.pdf')
    #     plt.show()  
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

def get_2d_translational_green(green):
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
    Green_temp = np.zeros((N,N),dtype=complex)
    for i in range(1,N,2):
        for j in range(N):
            if j - i in even_green_dict:
                Green_temp[i,j] = even_green_dict[j-i]
    for i in range(0,N,2):
        for j in range(N):
            if j - i in odd_green_dict:
                Green_temp[i,j] = odd_green_dict[j-i]
    return Green_temp

clist = ['purple','red','green','blue','black','y','r','g','b']
# clist = ['red','green','blue','black']
style_list = [':','-','-.','--','+','-','-','-','-']
# style_list = ['-','-.','--','+']

fig, axs = plt.subplots(2, 1,  figsize=(8,10))
residual_list = []
start_time_list = []
# file_name = '8th_results/U8_dag/'
# for files in ['2th_results','4th_results','8th_results','half_results']:
#     for case in ['U4','U4_dag']:
for files in ['half_results']:
    for case in ['U8']:
        file_name = files+'/'+case+'/'
        muti_num = 6
        interval_step = 5
        # start_time = 100
        start_time = 60
        # f = open('figure_data/{}_{}_mul_{}_int_{}_stability_t10_M1.txt'.format(files, case, muti_num, interval_step),'w')
        # muti_num_grid = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        # muti_num_grid = [1,2,3,4,5,6,7,8,9,10]
        # muti_num_grid = [i for i in range(1,muti_num*10)]
        muti_num_grid = [1,5,10,15]
        # muti_num_grid = [10]
        train_range = 80
        # interval_step_grid = [5]
        # interval_step_grid = [1,2,4,8,12,16]
        interval_step_grid = [1,2,3,4,5,6,7,8,9,10]
        res_grid = np.zeros([len(muti_num_grid), len(interval_step_grid)])
        dis_grid = np.zeros([len(muti_num_grid), len(interval_step_grid)])
        # res_grid = np.zeros([151, len(interval_step_grid)])
        # dis_grid = np.zeros([151, len(interval_step_grid)])
        res_step_list = []
        dis_step_list = []
        m_idx = 0
        # for start_time in [30,40,50,60,70,80]:
        # for start_time in [60]:
        for m_idx, muti_num in enumerate(muti_num_grid):
            c_idx = m_idx
        # for c_idx, train_range in zip([0,1,2,3],[20,40,60,80]):
        # for c_idx, train_range in zip([0,1,2,3,4,5,6,7,8],[10,20,30,40,50,60,70,80,90]):
        # for c_idx, train_range in zip([0,1,2],[40,60,80]):
            for i_idx, interval_step in enumerate(interval_step_grid):
                # m_idx = int(start_time/interval_step)
                # muti_num = int(train_range/interval_step) + 1
                recursion_step = start_time - (muti_num-1)*interval_step
                start_time_list.append(recursion_step)
                if recursion_step <= 0:
                    res_grid[m_idx, i_idx] = res_grid[0, 0]
                    dis_grid[m_idx, i_idx] = dis_grid[0, 0]
                    continue

                tlimit_compare = 150
                # tlimit = 600
                tlimit = 480
                extend_N = 0
                Nreal = 128
                factor = 1
                Green_tn = np.zeros((tlimit,Nreal*factor,Nreal*factor),dtype=complex)
                re_Green_tn = np.zeros((tlimit,Nreal*factor,Nreal*factor),dtype=complex)
                Green_original_tn = np.zeros((tlimit_compare+1,Nreal*factor,Nreal*factor),dtype=complex)
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
                residual = np.zeros((tlimit_compare+1,Nreal*factor,Nreal*factor),dtype=complex)
                new_residual = np.zeros((tlimit_compare+1,Nreal*factor,Nreal*factor),dtype=complex)
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
                # for t in range(tlimit_compare):
                #     residual_list_dict[t] = residual[t]
                #     green_list_dict[t] = residual[t]  
                # plt.plot([t*0.1 for t in range(tlimit_compare)], [Green_original_tn[t][int(N/2)-1,int(N/2)-1] for t in range(tlimit_compare)],label='dmrg')        
                
                for j in range(t_len):
                    t1 = recursion_step+j*interval_step
                    print('t1: ',t1)
                    # Ci[t1] = np.matmul(overlap_pinv, residual[t1].T)        # wrong way to get Ci
                    Ci[t1] = np.matmul(residual[t1], overlap_pinv)        # this should be full

                    # w, v = LA.eig(Ci[t1])
                    # for kval in w:
                    #     if np.abs(kval) > 1.0:
                    #         print(kval)
                    # plt.plot([t*0.1 for t in range(t1, tlimit_compare)], [residual[t][int(N/2)-1,int(N/2)-1] for t in range(t1, tlimit_compare)],label=str(t1))

                    for t in range(t1, tlimit_compare):
                    # for t in range(t1, tlimit):
                        new_residual[t] = residual[t] - np.matmul(Ci[t1], Green_original_tn[t-t1])        # this should have weight
                        residual_list_dict[t] = new_residual[t]

                        cum_green_list[t] += np.matmul(Ci[t1], Green_original_tn[t-t1])
                        # green_list_dict[t] = np.matmul(Ci[t1], Green_original_tn[t-t1])
                    # plt.plot([t*0.1 for t in range(t1, tlimit_compare)], [new_residual[t][int(N/2)-1,int(N/2)-1] for t in range(t1, tlimit_compare)],label='after projection '+str(t1))
                    # plt.plot([t*0.1 for t in range(t1, tlimit_compare)], [cum_green_list[t][int(N/2)-1,int(N/2)-1] for t in range(t1, tlimit_compare)],label='after projection '+str(int(0.1*t1)))

                    t1_list.append(t1*0.1)
                    residual_dict[t1] = new_residual[t1]
                    residual= new_residual.copy()
                # plt.plot([t*0.1 for t in range(t1, tlimit_compare)], [new_residual[t][int(N/2)-1,int(N/2)-1] for t in range(t1, tlimit_compare)],label=str(muti_num))
                # plt.legend()
                # plt.savefig('./new_multi_projection/{}_multi_improve_residual_same_point.pdf'.format('_'.join(file_name.split('/'))))
                # plt.show()
                # raise SystemExit(0)

                # plt.axhline(y=0,c='r')
                # # plt.plot([t for t in sorted(green_list_dict.keys())], [green_list_dict[t][int(N/2)-1,int(N/2)-1] for t in sorted(residual_list_dict.keys())])
                # plt.plot([t for t in sorted(residual_list_dict.keys())], [residual_list_dict[t][int(N/2)-1,int(N/2)-1] for t in sorted(residual_list_dict.keys())])
                # # plt.plot(t1_list, [residual_dict[t][int(N/2)-1,int(N/2)-1] for t in range(recursion_step, recursion_step+t_len)],'r-.',label='residual')
                # plt.show()
                # raise SystemExit(0)
                
                # plt.tick_params(
                #     axis='x',          # changes apply to the x-axis
                #     which='both',      # both major and minor ticks are affected
                #     bottom=False,      # ticks along the bottom edge are off
                #     left=False,         # ticks along the top edge are off
                #     labelbottom=True) # labels along the bottom edge are off
                # axs[1].plot([t*0.1 for t in range(t1, tlimit_compare)], [-1.0*new_residual[t][int(N/2)-1,int(N/2)-1] for t in range(t1, tlimit_compare)], style_list[c_idx], c=clist[c_idx], label='Range '+str(int(0.1*(start_time - recursion_step))))
                # axs[1].plot([t*0.1 for t in range(t1, tlimit_compare)], [-1.0*new_residual[t][int(N/2)-1,int(N/2)-1] for t in range(t1, tlimit_compare)], style_list[c_idx], c=clist[c_idx], label='M = '+str(muti_num))

                print(residual_dict.keys())
                print(Ci.keys())
                # for tm in Ci:
                #     plt.figure()
                #     plt.plot(Ci[tm][int(N/2),],label=str(tm))
                #     plt.savefig('./multi_projection/Ci_plot/{}_Gaussian_20_norm1_Ci_tm{}_plot.pdf'.format('_'.join(file_name.split('/')),tm ))
                    # plt.show()
                # raise SystemExit(0)

                green_list = dict()
                green_cum_list = dict()
                for tm in range(recursion_step,recursion_step+t_len):
                    green_list[tm] = dict()
                    green_cum_list[tm] = dict()
                    for t in range(recursion_step+t_len):
                        green_cum_list[tm][t] = re_Green_tn[t].copy()
                        green_list[tm][t] = re_Green_tn[t].copy()

                for t in range(recursion_step+((t_len)-1)*interval_step,tlimit):
                    for j in range(t_len):
                    # for j in range(12):
                        tm = recursion_step+j*interval_step
                        if t-tm >= 0:
                            re_Green_tn[t] += np.matmul(Ci[tm],re_Green_tn[t-tm])
                        
                cgi = 0.0
                for tm in Ci:
                    for i in range(N):
                        cgi += np.conj(Ci[tm][int(N/2),i])*Green_tn[tm][int(N/2),i]
                        cgi += Ci[tm][int(N/2),i]*np.conj(Green_tn[tm][int(N/2),i])
                print('original cgi: ',cgi)

                ccgi = 0.0
                for tm in Ci:
                    for i in range(N):
                        for j in range(N):
                            ccgi += Ci[tm][int(N/2),i]*np.conj(Ci[tm][int(N/2),j])*overlap[i][j]
                print('original ccgi: ',ccgi)
                print('original residual: ', overlap[int(N/2)][int(N/2)] - cgi + ccgi)

                cgi = 0.0
                for tm in Ci:
                    cgi += 2*np.real(np.dot( np.conj(Ci[tm][int(N/2),:]), Green_tn[tm][int(N/2),:] ))
                    # cgi += np.dot( Ci[tm][int(N/2),:], np.conj(Green_tn[tm][int(N/2),:]) )
                print('cgi: ',cgi)

                ccgi = 0.0
                for tm in Ci:
                    for tn in Ci:
                        if tm >= tn:
                            ccgi += np.dot( np.conj(Ci[tm][int(N/2),:]), np.matmul( Green_tn[abs(tm-tn)], Ci[tn][int(N/2),:]  ) )
                        else:
                            ccgi += np.dot( np.conj(Ci[tm][int(N/2),:]), np.matmul( np.conj(Green_tn[abs(tm-tn)]), Ci[tn][int(N/2),:]  ) )
                print('ccgi: ',ccgi)

                residual = overlap[int(N/2)][int(N/2)] - cgi + ccgi
                print('residual: ', overlap[int(N/2)][int(N/2)] - cgi + ccgi)
                res_grid[m_idx, i_idx] = residual

                residual_list.append(residual)

                for i in range(tlimit_compare+1):
                    t = 0.1*i
                    N601_U1_t = get_Green_snapshot_at_center('./{}/Green_sites'.format(file_name),t,N=Nreal)
                    real_Green_center_tn[i,:] = N601_U1_t

                # import os
                # green_save_dir = '{}/Green_multi_re_t{}'.format(file_name, start_time)
                # if not os.path.isdir(green_save_dir):
                #     os.mkdir(green_save_dir)

                # for t in range(tlimit):
                # # #     # with open('./results/smooth_center_compare/N{}_smooth_N{}_Green_matrix_t{}/Green_t{:.3f}.txt'.format(Nreal, Nshort, rstep, t*0.1),'w') as f:
                #     # with open('./N400_Hubbard_U2/Green_matrix_rstep{}/Green_t{:.3f}.txt'.format(int(rstep*0.1), t*0.1),'w') as f:
                #     with open('{}/Green_t{:.3f}.txt'.format(green_save_dir, t*0.1),'w') as f:
                #         for i in range(N):
                #             # f.write(str(i+1)+ ' ' + str(np.real(Green_tn[t,int(N/2)-1,i])) + ' ' + str(np.imag(Green_tn[t,int(N/2)-1,i])) + '\n' )
                #             f.write(str(np.real(re_Green_tn[t,int(N/2)-1,i])) + ' ' + str(np.imag(re_Green_tn[t,int(N/2)-1,i])) + '\n' )

                # plt.axvline(x=recursion_step*0.05,ls='--',c='green')
                center_evolve = np.array([re_Green_tn[i,int(N/2)-1,int(N/2)-1]*1.0 for i in range(tlimit)])
                real_Green_center = np.array([real_Green_center_tn[i,int(N/2)-1] for i in range(len(real_Green_center_tn))])

                distance = np.sum(np.abs(real_Green_center[start_time:tlimit_compare]-center_evolve[start_time:tlimit_compare]))
                distance = distance/(tlimit_compare-start_time)
                dis_grid[m_idx, i_idx] = distance/(tlimit_compare-start_time)

                t = [i*0.1 for i in range(len(real_Green_center_tn))]
                # if muti_num == 2:
                    # plt.plot(t, real_Green_center,'-',color='black',label=file_name)
                # if c_idx == 0:
                #     axs[0].plot(t, real_Green_center,'-',color='black', label='DMRG')

                # plt.figure()
                t = [i*0.1*1.0 for i in range(tlimit)]
                plt.plot(t, center_evolve,'-',color='red',label='Recursion for t > {}'.format(int(recursion_step*0.1)))
                # plt.plot(t, center_evolve,'-',label='Recursion for t > {} spaceing = {}'.format(int(recursion_step*0.1), interval_step))
                # axs[0].plot(t, center_evolve, style_list[c_idx], color=clist[c_idx], label='M = {}'.format(str(muti_num)))

                # if interval_step == 1:
                    # plt.axvline(x=0.1*recursion_step,ls='--',c='g')
                    # plt.axvline(x=0.1*(recursion_step+(muti_num-1)*interval_step),ls='--',c='g')
                    # axs[0].axvline(x=0.1*recursion_step,ls='--',c='g')
                    # axs[0].axvline(x=0.1*(recursion_step+(muti_num-1)*interval_step),ls='--',c='g')
                
                # plt.tick_params(direction='in',labelsize=12)

                # plt.title('mult:{}, interval:{}, res:{:.3f}, dis:{:.3f} '.format(muti_num, interval_step, np.real(residual), distance))
                # plt.locator_params(nbins=6,axis='y')
                # plt.legend(fontsize=16)
                # plt.xlabel('t',fontsize=16)
                # plt.ylabel("$ReC^{\dagger}C(x=0,t)$",fontsize=16)
                # plt.ylabel("$ReG(x=0,t)$",fontsize=16)
                # plt.tight_layout()   
                # plt.savefig('./new_multi_projection/{}_mul{}_int{}_start_{}_long_run.pdf'.format('_'.join(file_name.split('/')), muti_num, interval_step, start_time))
                
                res_step_list.append(np.real(residual))
                dis_step_list.append(np.real(distance))
                # plt.show()

                # f.write(str(muti_num)+'\n')
                # f.write(' '.join([str(t) for t in [i*0.1*1.0 for i in range(tlimit)]])+'\n')
                # f.write(' '.join([str(v) for v in center_evolve])+'\n') 
    
        # axins3 = inset_axes(axs[1],
        #                     width="30%", # width = 30% of parent_bbox
        #                     height=1., # height : 1 inch
        #                     loc=3)

        # axins3 = axs[1].inset_axes([0.14,0.12,0.4,0.4])
        # # axins3 = inset_axes(axs[1], width="100%", height="100%",
        # #                     bbox_to_anchor=(1e-2, 2, 1e3, 3))
        # print(residual_list)
        # # axins3.plot([0.1*(start_time - t) for t in start_time_list], [float(v) for v in residual_list])
        # axins3.plot([M for M in muti_num_grid], [float(v) for v in residual_list])
        # axins3.set_xlabel('M',fontsize=10)
        # axins3.set_ylabel('$<R|R>$',fontsize=10)
        # axins3.tick_params(direction='in',labelsize=11)
        # axins3.set_yticks(np.arange(0.5, 0.55, 0.025))

        # # axs[0].plot([t*0.1 for t in start_time_list],[real_Green_center[t] for t in start_time_list],'*',markersize=15)
        # # for t in start_time_list:
        # #     axs[0].axvline(x=t*0.1,ls='--',c='black')
        # for s_idx in range(len(start_time_list)):
        #     axs[0].axvline(x=start_time_list[s_idx]*0.1,ls='--',c=clist[s_idx])

        # # axs[0].axvline(x=start_time*0.1,ls='--',c='black')
        # axs[0].legend(fontsize=12)
        # axs[0].text(40,0.15, '(a)', fontsize=18)

        # axs[0].tick_params(direction='in',labelsize=16)
        # axs[0].locator_params(axis='y', nbins=6)
        # axs[0].set_xlabel('t',fontsize=16)
        # axs[0].set_ylabel("$ReG(x=0,t)$",fontsize=16)
        # axs[0].set_ylim([-0.2,0.2])
        
        # axs[1].text(13,0.02, '(b)', fontsize=18)
        # axs[1].tick_params(direction='in',labelsize=16)
        # axs[1].legend(fontsize=12)
        # axs[1].set_xlabel('t',fontsize=16)
        # axs[1].set_ylabel("$ReG(x=0,t)_{diff}$",fontsize=16)
        # plt.tight_layout()
        # # plt.savefig('./figure_data/multi_recursion_{}_study_vsM_vlines_t10_1.pdf'.format('_'.join(file_name.split('/'))))
        # plt.show()

    # f.write(str('DMRG')+'\n')
    # f.write(' '.join([str(t) for t in [i*0.1 for i in range(len(real_Green_center_tn))]])+'\n')
    # f.write(' '.join([str(v) for v in real_Green_center])+'\n') 

    # f.write(' '.join([str(np.real(v)) for v in residual_list])+'\n') 

    # # plt.figure()
    # plt.title('dir: {}, train_range: {}'.format('_'.join(file_name.split('/')), train_range))

    # plt.plot([30,40,50,60,70,80], dis_step_list,label='distance')
    # plt.plot(interval_step_grid, res_step_list,label='residual')
    # plt.savefig('./new_multi_projection/stability/{}_change_spacing_multi_10_compare_list.pdf'.format('_'.join(file_name.split('/')) ))
    # plt.show()
    
    # print(residual_list)
    # plt.plot([4,8,12],residual_list)
    plt.show()

    # f.close()
    # plt.figure()
    # plt.imshow(dis_grid)
    # plt.ylabel('projections')
    # plt.xlabel('interval')
    # plt.colorbar()
    # # plt.savefig('./new_multi_projection/{}_dis_grid.pdf'.format('_'.join(file_name.split('/'))))

    plt.figure()
    plt.imshow(dis_grid)
    # plt.imshow(res_grid)
    plt.text(2,2,'(b)',fontsize=16)
    plt.ylabel('M',fontsize=16)
    plt.xlabel('spacing x 0.1',fontsize=16)
    plt.colorbar()
    # plt.savefig('./new_multi_projection/{}_dis_grid_new_style.pdf'.format('_'.join(file_name.split('/'))))
