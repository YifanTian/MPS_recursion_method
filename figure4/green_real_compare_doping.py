import matplotlib.pyplot as plt
import numpy as np

case = 'U1'
for case in ['U1_dag','U2_dag','U4_dag','U8_dag']:
# for case in ['U8']:
    file_name = '4th_results'
    muti_num = 10
    interval_step = 5
    start_time = 60

    projection_time_list = [start_time-i*interval_step for i in range(muti_num)][::-1]

    # plt.figure(figsize=(8,8))
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6,6))
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)

    axs[0].axvline(x=0.1*start_time,ls='--',c='g')
    axs[1].axvline(x=0.1*start_time,ls='--',c='g')
    axs[2].axvline(x=0.1*start_time,ls='--',c='g')
    axs[0].axvline(x=0.1*(start_time-interval_step*(muti_num-1)),ls='--',c='g')
    axs[1].axvline(x=0.1*(start_time-interval_step*(muti_num-1)),ls='--',c='g')
    axs[2].axvline(x=0.1*(start_time-interval_step*(muti_num-1)),ls='--',c='g')

    axs[0].text(20,0.15,'p=1/4',fontsize=14)
    axs[1].text(20,0.15,'p=1/8',fontsize=14)
    axs[2].text(20,0.15,'p=0',fontsize=14)

    save_file_name = 'ladder_system_test_{}_{}__mul{}_int{}_start_{}_factor2_savefile.txt'.format(file_name, case, muti_num, interval_step, start_time)
    with open('{}'.format(save_file_name),'r') as f:  
        content = f.readlines()

    dmrg_t = [float(t) for t in content[0].split(' ')]
    dmrg_green = [np.complex(v) for v in content[1].strip().split(' ')]
    re_t = [float(t) for t in content[2].split(' ')]
    re_green = [np.complex(v) for v in content[3].strip().split(' ')]
    lp_t = [float(t) for t in content[4].split(' ')]
    lp_green = [np.complex(v) for v in content[5].strip().split(' ')]


    # Plot each graph, and manually set the y tick values
    l1, = axs[0].plot(dmrg_t, dmrg_green, 'black',label='DMRG')
    axs[0].plot([0.1*t for t in projection_time_list], [dmrg_green[t] for t in projection_time_list], '*', c='red')
    l2, = axs[0].plot(re_t, re_green,'red',label='Recursion')
    l3, = axs[0].plot(lp_t, lp_green,'blue',label='LP')
    # axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
    axs[0].set_yticks(np.arange(-0.2, 0.21, 0.2))
    axs[0].set_ylim(-0.2, 0.2)

    file_name = '8th_results'
    save_file_name = 'ladder_system_test_{}_{}__mul{}_int{}_start_{}_factor2_savefile.txt'.format(file_name, case, muti_num, interval_step, start_time)
    with open('{}'.format(save_file_name),'r') as f:  
        content = f.readlines()

    dmrg_t = [float(t) for t in content[0].split(' ')]
    dmrg_green = [np.complex(v) for v in content[1].strip().split(' ')]
    re_t = [float(t) for t in content[2].split(' ')]
    re_green = [np.complex(v) for v in content[3].strip().split(' ')]
    lp_t = [float(t) for t in content[4].split(' ')]
    lp_green = [np.complex(v) for v in content[5].strip().split(' ')]

    axs[1].plot(dmrg_t, dmrg_green, 'black')
    axs[1].plot([0.1*t for t in projection_time_list], [dmrg_green[t] for t in projection_time_list], '*', c='red')
    axs[1].plot(re_t, re_green, 'red')
    axs[1].plot(lp_t, lp_green, 'blue')
    axs[1].set_yticks(np.arange(-0.2, 0.2, 0.2))
    axs[1].set_ylim(-0.2, 0.2)

    file_name = 'half_results'
    save_file_name = 'ladder_system_test_{}_{}__mul{}_int{}_start_{}_factor2_savefile.txt'.format(file_name, case, muti_num, interval_step, start_time)
    with open('{}'.format(save_file_name),'r') as f:  
        content = f.readlines()

    dmrg_t = [float(t) for t in content[0].split(' ')]
    dmrg_green = [np.complex(v) for v in content[1].strip().split(' ')]
    re_t = [float(t) for t in content[2].split(' ')]
    re_green = [np.complex(v) for v in content[3].strip().split(' ')]
    lp_t = [float(t) for t in content[4].split(' ')]
    lp_green = [np.complex(v) for v in content[5].strip().split(' ')]

    # plt.axvline(x=recursion_step*0.05,ls='--',c='green')
    # axs[2].axvline(x=0.1*start_time,ls='--',c='g')

    axs[2].plot(dmrg_t, dmrg_green, 'black')
    axs[2].plot([0.1*t for t in projection_time_list], [dmrg_green[t] for t in projection_time_list], '*', c='red')
    axs[2].plot(re_t, re_green, 'red')
    axs[2].plot(lp_t, lp_green, 'blue')
    axs[2].set_yticks(np.arange(-0.2, 0.2, 0.2))
    axs[2].set_ylim(-0.2, 0.2)

    axs[1].set_ylabel("$ReG(x=0,t)$",fontsize=16)
    axs[2].set_xlabel('t',fontsize=16)
    axs[0].tick_params(direction='in',labelsize=12)
    axs[1].tick_params(direction='in',labelsize=12)
    axs[2].tick_params(direction='in',labelsize=12)


    fig.legend((l1, l2, l3), ('DMRG', 'Recursion', 'LP'), ncol=3, loc='upper center', fontsize=14)
    # plt.legend()
    # fig.legend(ncol=3, bbox_to_anchor=(0, 1),
                #   loc='lower left', fontsize='small')
    plt.savefig('{}_mul{}_int{}_start_{}_doping.pdf'.format(case, muti_num, interval_step, start_time))
    # plt.show()