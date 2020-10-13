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
        print(y[ii])
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


# def overlap_mat(avg0, avg1, avg2, avg3, avg4, m):
def overlap_mat(all_avg, m):
    mn = len(all_avg)
    # mat = np.zeros([5*m, 5*m])
    mat = np.zeros([mn*m, mn*m],dtype=complex)
    # all_avg = [avg1, avg2, avg3, avg4]
    # all_avg = [avg1, avg2, avg3, avg4]
    # avg0 = all_avg[0]
    # all_avg = all_avg[1:]

    for k in range(0,mn):
        for l in range(0,mn):
            print(k,l)
            if l >= k:
                mat[k*m:(k+1)*m,l*m:(l+1)*m] = np.conj(all_avg[l-k])
            else:
                mat[k*m:(k+1)*m,l*m:(l+1)*m] = all_avg[abs(l-k)]
    return mat

def overlap_neg_mat(all_avg, projection_points, m):
    mn = len(projection_points)
    # mat = np.zeros([5*m, 5*m])
    mat = np.zeros([mn*m, mn*m],dtype=complex)
    # all_avg = [avg1, avg2, avg3, avg4]
    # all_avg = [avg1, avg2, avg3, avg4]
    # avg0 = all_avg[0]
    # all_avg = all_avg[1:]

    # for k in range(0,mn):
    #     for l in range(0,mn):
    for k,k_val in enumerate(projection_points):
        for l,l_val in enumerate(projection_points):
            print(k, l, k_val, l_val)
            if l_val >= k_val:
                mat[k*m:(k+1)*m,l*m:(l+1)*m] = np.conj(all_avg[l_val-k_val])
            else:
                mat[k*m:(k+1)*m,l*m:(l+1)*m] = all_avg[abs(l_val-k_val)]
    return mat


tlimit_compare = 150
tlimit = 240

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--doping', type=str,
                     help='doping',default='8th')
parser.add_argument('--start_time', type=int,
                     help='start_time',default=60)
parser.add_argument('--U', type=int,
                     help='U',default=4)
args = parser.parse_args()

style_list = ['-','--','-','--','-']
# file_name = '8th_results/U8_dag/'
# for files in ['2th_results','4th_results','8th_results','half_results']:
#     for case in ['U4','U4_dag']:
# new_dir_name = './4th_results'
for files in ['../ladder_system_test/{}_results'.format(args.doping)]:
    new_dir_name = files.split('/')[-1]
    # for case in ['U{}'.format(args.U),'U{}_dag'.format(args.U)]:
    for case in ['U{}_dag'.format(args.U)]:
        file_name = files+'/'+case+'/'
        muti_num = 6
        interval_step = 5
        # start_time = 150
        start_time = args.start_time
        # muti_num_grid = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        # muti_num_grid = [i for i in range(1,muti_num*10)]
        # muti_num_grid = [2,4,6,8,10,12]
        # muti_num_grid = [5]
        # muti_num_grid = [2,3,4,5,6,7,8,9,10]
        # muti_num_grid = [1,3]
        muti_num_grid = [3]
        # interval_step_grid = [1,2,3,4,5,6,7,8,9,10]
        interval_step_grid = [1]
        # projection_points = [0,5,10,15,20,25]
        # projection_points_list = [[0,1,2],[0,1],[-1,0,1],[-1,0,1,2],[-2,-1,0,1,2]]
        # projection_points_list = [[0,1,2],[0,1],[-1,0,1],[-1,0,1,2],[-2,-1,0,1,2]]
        projection_points_list = [[0,1,2],[0,1],[0,1,2,4],[0,1,2,4,8,16,32]]

        # res_grid = np.zeros([len(muti_num_grid), len(interval_step_grid)])
        # dis_grid = np.zeros([len(muti_num_grid), len(interval_step_grid)])
        res_grid = np.zeros([151, len(projection_points_list)])
        dis_grid = np.zeros([151, len(projection_points_list)])
        res_step_list = []
        dis_step_list = []
        m_idx = 0
        plt.figure()
        # for start_time in [30,40,50,60,70,80]:
        # for start_time in [60]:
        for m_idx, muti_num in enumerate(muti_num_grid):
            # for i_idx, interval_step in enumerate(interval_step_grid):
            for i_idx, projection_points in enumerate(projection_points_list):
                interval_step = interval_step_grid[0]
                # m_idx = int(start_time/interval_step)
                # muti_num = m_idx
                # projection_points = [interval_step*i for i in range(muti_num)]
                # projection_points = [-2,-1,0,1,2]
                # projection_points = [0,1,2]
                recursion_step = start_time - (muti_num-1)*interval_step
                print()
                if recursion_step <= 0:
                    res_grid[m_idx, i_idx] = res_grid[0, 0]
                    dis_grid[m_idx, i_idx] = dis_grid[0, 0]
                    continue

                tlimit_compare = 150
                tlimit = 240
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

                # overlap_larger = overlap_mat(avg0, avg10, avg20, avg30, avg40, N)
                # overlap_larger = overlap_mat([Green_original_tn[t] for t in projection_points], N)
                overlap_larger = overlap_neg_mat(Green_original_tn, projection_points, N)
                # ============================================
                # plt.plot(np.imag(overlap_larger[int(N/2),]))
                # plt.show()
                # raise SystemExit(0)

                N = Nreal
                overlap = Green_tn[0]

                overlap_pinv = np.linalg.pinv(overlap,rcond=1e-4)

                lb_Ci = dict()
                overlap_larger_pinv = np.linalg.pinv(overlap_larger,rcond=1e-4)
                print(overlap_larger_pinv.shape)
                gt = np.concatenate([Green_tn[start_time-t] for t in projection_points],axis=1)
                print(gt.shape)

                # plt.plot(overlap_larger_pinv[int(N/2),])
                # plt.plot(overlap_pinv[int(N/2),])
                # plt.show()
                # raise SystemExit(0)

                ct = np.matmul(gt, overlap_larger_pinv)
                # ct = np.matmul(gt, overlap_pinv)
                print(ct.shape)

                restore_gt = np.matmul(ct, overlap_larger[:,0:N])
                Green_tn_t60_restore = restore_gt[:,0:N]

                # print(overlap_larger[:,0:N].shape)
                # plt.plot(np.imag(overlap_larger[:,0:N][N:2*N,:][int(N/2),]),'-')
                # plt.plot(np.imag(Green_original_tn[10][int(N/2),]),'*')
                # plt.show()
                # raise SystemExit(0)

                # restore_gt_t60 = np.matmul(ct, np.concatenate([Green_original_tn[0], Green_original_tn[10], Green_original_tn[20],\
                # Green_original_tn[30], Green_original_tn[40]],axis=0))

                # plt.plot(Green_tn_t60_restore[int(N/2),])
                # plt.plot(Green_tn[60][int(N/2),])
                # plt.plot(restore_gt_t60[int(N/2),])
                # plt.show()
                # raise SystemExit(0)

                for idx, t in enumerate(projection_points):
                    lb_Ci[start_time - t] = ct[:,idx*N:(idx+1)*N]
                print(lb_Ci.keys())
                    
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

                print(residual_dict.keys())
                print(Ci.keys())
                # for tm in Ci:
                #     plt.figure()
                #     plt.plot(Ci[tm][int(N/2),],label=str(tm))
                #     plt.savefig('./multi_projection/Ci_plot/{}_Gaussian_20_norm1_Ci_tm{}_plot.pdf'.format('_'.join(file_name.split('/')),tm ))
                    # plt.show()

                green_list = dict()
                green_cum_list = dict()
                for tm in range(recursion_step,recursion_step+t_len):
                    green_list[tm] = dict()
                    green_cum_list[tm] = dict()
                    for t in range(recursion_step+t_len):
                        green_cum_list[tm][t] = re_Green_tn[t].copy()
                        green_list[tm][t] = re_Green_tn[t].copy()


                lre_Green_tn = np.zeros((tlimit,Nreal*factor,Nreal*factor),dtype=complex)
                for i in range(max(list(lb_Ci.keys()))):
                    lre_Green_tn[i,:,:] = Green_tn[i,:,:]
                # plt.axvline(x = 60*0.1, c='g')
                # for t in range(recursion_step+((t_len)-1)*interval_step,tlimit):
    
                for t in range(start_time,tlimit):
                    for tm in Ci:
                        if t-tm >= 0:
                            re_Green_tn[t] += np.matmul(Ci[tm],re_Green_tn[t-tm])

                # for t in range(start_time,tlimit):
                for t in range(max(list(lb_Ci.keys())),tlimit):
                    for tm in lb_Ci:
                        if t-tm >= 0:
                            lre_Green_tn[t] += np.matmul(lb_Ci[tm],lre_Green_tn[t-tm])
                        
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

                for i in range(tlimit_compare+1):
                    t = 0.1*i
                    N601_U1_t = get_Green_snapshot_at_center('./{}/Green_sites'.format(file_name),t,N=Nreal)
                    real_Green_center_tn[i,:] = N601_U1_t

                # import os
                # green_save_dir = '{}/{}/Green_multi_re_t{}'.format(new_dir_name, case, start_time)
                # if not os.path.isdir(green_save_dir):
                #     os.mkdir(green_save_dir)

                # for t in range(tlimit):
                # # #     # with open('./results/smooth_center_compare/N{}_smooth_N{}_Green_matrix_t{}/Green_t{:.3f}.txt'.format(Nreal, Nshort, rstep, t*0.1),'w') as f:
                #     # with open('./N400_Hubbard_U2/Green_matrix_rstep{}/Green_t{:.3f}.txt'.format(int(rstep*0.1), t*0.1),'w') as f:
                #     with open('{}/Green_t{:.3f}.txt'.format(green_save_dir, t*0.1),'w') as f:
                #         for i in range(N):
                #             # f.write(str(i+1)+ ' ' + str(np.real(Green_tn[t,int(N/2)-1,i])) + ' ' + str(np.imag(Green_tn[t,int(N/2)-1,i])) + '\n' )
                            # f.write(str(np.real(re_Green_tn[t,int(N/2)-1,i])) + ' ' + str(np.imag(re_Green_tn[t,int(N/2)-1,i])) + '\n' )

                # plt.axvline(x=recursion_step*0.05,ls='--',c='green')
                center_evolve = np.array([re_Green_tn[i,int(N/2)-1,int(N/2)-1]*1.0 for i in range(tlimit)])
                real_Green_center = np.array([real_Green_center_tn[i,int(N/2)-1] for i in range(len(real_Green_center_tn))])

                distance = np.sum(np.abs(real_Green_center[start_time:tlimit_compare]-center_evolve[start_time:tlimit_compare]))
                distance = distance/(tlimit_compare-start_time)
                dis_grid[m_idx, i_idx] = distance/(tlimit_compare-start_time)

                # plt.figure()
                re_t = [i*0.1*1.0 for i in range(tlimit)]
                # plt.plot(re_t, center_evolve,'-',color='red',label='Recursion for t > {}'.format(int(start_time*0.1)))
                if i_idx == 0:
                    # plt.plot(re_t[start_time:len(real_Green_center)], center_evolve[start_time:len(real_Green_center)] - real_Green_center[start_time:len(real_Green_center)],'--',label='Multistep for t > {}'.format(int(start_time*0.1)))
                    plt.plot(re_t[start_time:len(real_Green_center)], center_evolve[start_time:len(real_Green_center)] - real_Green_center[start_time:len(real_Green_center)],'--',label='Multistep M = 3')

                # plt.plot(t, center_evolve,'-',label='Recursion for t > {}'.format(int(recursion_step*0.1)))
                dmrg_t = [i*0.1 for i in range(len(real_Green_center_tn))]
                # if muti_num == 2:
                # plt.plot(t, real_Green_center,'-',color='black',label=file_name)
                # plt.plot(dmrg_t, real_Green_center,'-',color='black',label='DMRG')

                # plt.plot([t*0.1 for t in projection_points], [real_Green_center[t] for t in projection_points], '*', c='r')

                lre_center_evolve = np.array([lre_Green_tn[i,int(N/2)-1,int(N/2)-1]*1.0 for i in range(tlimit)])
                # plt.plot(re_t, lre_center_evolve,'-',color='blue',label='Expend Basis Recursion for t > {} N: {}'.format(int(start_time*0.1), muti_num))

                # plt.plot(re_t[max(list(lb_Ci.keys())):len(real_Green_center)],lre_center_evolve[max(list(lb_Ci.keys())):len(real_Green_center)]-real_Green_center[max(list(lb_Ci.keys())):len(real_Green_center)], label='Expend Basis Recursion for t > {} N: {}'.format(int(start_time*0.1), muti_num))
                plt.plot(re_t[max(list(lb_Ci.keys())):len(real_Green_center)],lre_center_evolve[max(list(lb_Ci.keys())):len(real_Green_center)]-real_Green_center[max(list(lb_Ci.keys())):len(real_Green_center)], style_list[i_idx],label='{}'.format(str([0.1*i for i in projection_points])))

                plt.xlim([6,7])
                lp_order = 5
                extend_N = tlimit - start_time
                Green_lp_real_list = list(burg_interpolation(real_Green_center[0:start_time],lp_order,extend_N))
                # plt.plot([t*0.1 for t in range(len(Green_lp_real_list))],  Green_lp_real_list, c = 'blue', label='LP')

                # save_file_name = '{}_mul{}_int{}_start_{}_savefile.txt'.format('_'.join(file_name.split('/')[1:]), muti_num, interval_step, start_time)
                # def save_data(save_file_name, dmrg, recursion, lp):
                #     with open('Green_extrapolation/{}'.format(save_file_name),'w') as f:  
                #         f.write(' '.join([str(t) for t in dmrg[0]])+'\n')
                #         f.write(' '.join([str(v) for v in dmrg[1]])+'\n')
                #         f.write(' '.join([str(t) for t in recursion[0]])+'\n')
                #         f.write(' '.join([str(v) for v in recursion[1]])+'\n')    
                #         f.write(' '.join([str(t) for t in lp[0]])+'\n')
                #         f.write(' '.join([str(v) for v in lp[1]])+'\n')     
                #     return 
                # save_data(save_file_name, [dmrg_t, real_Green_center], [re_t, center_evolve], [[t*0.1 for t in range(len(Green_lp_real_list))], Green_lp_real_list])

                if muti_num == 10:
                    plt.axvline(x=0.1*recursion_step,ls='--',c='g')
                    plt.axvline(x=0.1*(recursion_step+(muti_num-1)*interval_step),ls='--',c='g')

                plt.tick_params(direction='in',labelsize=12)
                # plt.title('mult:{}, interval:{}, res:{:.3f}, dis:{:.3f} '.format(muti_num, interval_step, np.real(residual), distance))
                plt.locator_params(nbins=6,axis='y')
                plt.legend(fontsize=12,loc='best')
                # plt.legend()
                plt.xlabel('t',fontsize=16)
                # plt.ylabel("$ReC^{\dagger}C(x=0,t)$",fontsize=16)
                plt.ylabel("$ReG(x=0,t)_{error}$",fontsize=16)
                plt.ylim([-0.015,0.02])
                plt.tight_layout()   

                res_step_list.append(np.real(residual))
                dis_step_list.append(np.real(distance))

            plt.text(6.8,-0.01,'(a)',fontsize=16)    
            # plt.savefig('./figures/multi_time_basis/Expand_basis_{}_mul{}_int{}_start_{}_neg_compare_error_points2_lines1.pdf'.format('_'.join(file_name.split('/')[2:]), muti_num, interval_step, start_time))
            plt.show()

        # plt.figure()
        # plt.title('dir: {}'.format('_'.join(file_name.split('/'))))
        # plt.plot([30,40,50,60,70,80], dis_step_list,label='distance')
        # plt.plot(interval_step_grid, res_step_list,label='residual')
        # plt.savefig('./new_multi_projection/stability/{}_start_time_compare_list.pdf'.format('_'.join(file_name.split('/'))))
        # plt.savefig('./Green_extrapolation/{}_mul{}_int{}_start_{}_long_run.pdf'.format('_'.join(file_name.split('/')[1:]), muti_num, interval_step, start_time))
        # plt.show()

    # plt.figure()
    # plt.imshow(dis_grid)
    # plt.ylabel('projections')
    # plt.xlabel('interval')
    # plt.colorbar()
    # # plt.savefig('./new_multi_projection/{}_dis_grid.pdf'.format('_'.join(file_name.split('/'))))

    # plt.figure()
    # plt.imshow(res_grid)
    # plt.ylabel('projections')
    # plt.xlabel('interval')
    # plt.colorbar()
    # # plt.savefig('./new_multi_projection/{}_res_grid.pdf'.format('_'.join(file_name.split('/'))))
