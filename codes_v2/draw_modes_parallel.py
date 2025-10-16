import numpy as np
from einops import rearrange

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
cmap = mpl.colors.LinearSegmentedColormap.from_list("ezmap", ["tab:blue","darkblue","cyan","green",'gold',"darkred","tab:red"])

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


from mpi4py import MPI
import os, sys

sys.path.append("/gpfs/users/botezv/.venv")
from SFEMaNS_env.SFEMaNS_object import define_mesh
from SFEMaNS_env.read_stb import get_fourier, get_mesh_gauss
from SFEMaNS_env.operators import nodes_to_gauss
##########################################
##########################################
##########################################

path_pod_computations = "/gpfs/users/botezv/APPLICATIONS_POD/pod_computations/codes/"
field = 'u'
# data_file = f"/gpfs/workdir/botezv/MY_APPLICATIONS_SFEMaNS_GIT/LES_VKS_TM73/RUNS/Runs_SFeMANS/Re1500_Rm150_mu50/data_tgcc/nonlin/PODs_out/test_manual/data_pod_test_manual.txt"
# data_file = f"/gpfs/workdir/botezv/MY_APPLICATIONS_SFEMaNS_GIT/LES_VKS_TM73/RUNS/Runs_SFeMANS/Re1500_Rm150_mu50/data_tgcc/nonlin/PODs_out/test_cross_cor/data_pod_test_cross_cor.txt"

data_file = f"/gpfs/workdir/botezv/MY_APPLICATIONS_SFEMaNS_GIT/LES_VKS_TM73/RUNS/Runs_SFeMANS/VKS_nst/Re500/data_pod_mean_field_penal_div.txt"
parallelize = False #DO NEVER CHANGE 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sys.path.append(path_pod_computations)

# from read_data import parameters
from initialization import init

##########################################
##########################################
##########################################

par, sfem_par = init(data_file,parallelize = parallelize)
sfem_par.add_bins(par.complete_output_path+par.output_file_name+f"/phys_pod_modes/m000/",field=f"POD_{par.field}",D=par.D)


path_to_mesh, mesh_ext, mesh_type, S, D, MF = par.path_to_mesh, par.mesh_ext, par.mesh_type, sfem_par.S, par.D, par.MF
#R, Z, W = get_mesh_gauss(sfem_par)
mesh = define_mesh(path_to_mesh, mesh_type)
R, Z = mesh.R, mesh.Z
    
##########################################
##########################################
##########################################

field = par.field
path_to_suites = par.complete_output_path+par.output_file_name

##############################################
##############################################
##############################################

is_it_from_phys = True
if is_it_from_phys:
    file = 'phys'
else:
    file = 'fourier'


path_out = path_to_suites + "/post_pro/"
os.makedirs(path_out, exist_ok=True)

path_out = path_out+f'/{file}_pod_modes/'
os.makedirs(path_out, exist_ok=True)


m_families = [m_family.min() for m_family in par.list_m_families]
for m in m_families:
    os.makedirs(path_out+f'/m{m:03d}', exist_ok=True)

##############################################
##############################################
##############################################


def def_save_name(m,nP,mF,is_it_from_phys):
    if is_it_from_phys:
        if not (m is None):
            return f'nP_{nP:03d}_mF_{mF:03d}'
        else:
            return f'nP_{nP:03d}_mF_{mF:03d}'
    else:
        return f'mF_{mF:03d}_nP_{nP:03d}'
    
##############################################
##############################################
##############################################


should_we_draw_mean_field = False

should_we_plot_opposite = False # To be put in relation with the sign of the latents especially for non-zero average
                                # because pod modes are determined up to a sign
should_we_make_separate_plots = True #relevant for vector fields when we want to keep all with arrows or not
fraction_arrows = 600
width = 0.01 # !!!! this coeff is directly proportionnal to the arrows' width
length_arrows = 0.015 # !!!! this coeff is inversely propotionnal to the arrows in the output
scale_type='dots'
should_we_save_plots = True
coordinates_to_plot = [0,1,2] # relevant for D = 3 in the case when parameter from above is set to True


should_we_give_indep_list_of_mF_and_nP = False # SET TO TRUE IF DEFINE list_mF and list_nP INDEPENDANTLY, SET TO FALSE IF PREFER TO SET list_nP_mF directly (cf what is below)

# list_m = [0, 1]
# list_nP = [0,1,2,3]
# list_mF = [0,1,2,3]
raw_list_nP_mF = []
for nP in range(4,10):
    for mF in range(0,3):
        raw_list_nP_mF.append((None,nP,mF))


raw_list_nP_mF = []
#raw_list_nP_mF += [(0, 2*nP, 8*(nP+1)) for nP in range(7)] + [(0, 1+2*nP, 8*(nP+1)) for nP in range(7)]
#raw_list_nP_mF += [(0, nP, 0) for nP in range(6)]
#raw_list_nP_mF += [(1, nP, 1) for nP in range(6)]
#raw_list_nP_mF += [(2, nP, 2) for nP in range(6)]
#raw_list_nP_mF += [(3, nP, 3) for nP in range(6)]
#raw_list_nP_mF += [(4, nP, 4) for nP in range(6)]
for m in range(5):
    for n in([-8,0,8]):
        raw_list_nP_mF += [(m, nP, np.abs(m+n)) for nP in range(20)]
    
# raw_list_nP_mF = [(None,0,0),(None,1,0),(None,2,0),(None,3,0),(None,0,1),(None,1,1),(None,2,1),(None,3,1),(None,0,2),(None,1,2),(None,2,2),(None,3,2)]

if is_it_from_phys == False:
    list_m = [0]

if should_we_give_indep_list_of_mF_and_nP:
    raw_list_nP_mF = []
    for m in list_m:
        for nP in list_nP:
            for mF in list_mF:
                raw_list_nP_mF.append((m,nP,mF))

list_nP_mF = []
for i in range(rank,len(raw_list_nP_mF),size):
    list_nP_mF.append(raw_list_nP_mF[i])

if should_we_draw_mean_field:
    list_mean_field = []
    for i in range(rank,len(list_mF),size):
        list_mean_field.append(list_mF[i])


##############################################
##############################################
##############################################
size_labels = 14
##############################################
##############################################
##############################################

factor_plot = R.max()/Z.max()

for m,nP,mF in list_nP_mF:
    if par.D == 1:
        if mF == 0:
            fig,ax = plt.subplots(1,figsize=(4*factor_plot,4))
        else:
            fig,ax = plt.subplots(1,2,figsize=(8*factor_plot,4))
    elif should_we_make_separate_plots:
        if mF == 0:
            fig,ax = plt.subplots(3,figsize=(4*factor_plot,12))
        else:
            fig,ax = plt.subplots(3,2,figsize=(8*factor_plot,12))
    else:
        if mF == 0:
            fig,ax = plt.subplots(1,figsize=(4*factor_plot,4))
        else:
            fig,ax = plt.subplots(1,2,figsize=(8*factor_plot,4))
    if mF == 0:
        list_fourier_types = ['cos']
    elif mF != 0:
        list_fourier_types = ['cos','sin']

    print(f'doing {def_save_name(m,nP,mF,is_it_from_phys)}')
    sfem_par.add_bins(par.complete_output_path+par.output_file_name+f"/phys_pod_modes/m{m:03d}/",field=f"POD_{par.field}",D=par.D,replace=True)
    all_data = get_fourier(sfem_par,nP+1,from_gauss=par.read_from_gauss)[:, :, np.array([mF])]
    #if par.read_from_gauss == False:
    #    all_data = nodes_to_gauss(all_data, par)
    all_data = rearrange(all_data, "N (D a) MF -> D N a MF", a=2)[:, :, :, 0]
    if should_we_plot_opposite:
        all_data *= -1

    for i,axis in enumerate(list_fourier_types):
        if axis != 'sin' or mF != 0:
            spectrum_to_plot = all_data[:, :, i]#rearrange(all_data ," (d N) -> d N",d=D)
#            WEIGHTS = np.array([W for _ in range(D)]).reshape(-1)

            if par.D == 1:
                borne = np.max(np.abs(spectrum_to_plot))
                if mF != 0:
                    im = ax[i].tripcolor(R, Z, spectrum_to_plot[0, :], cmap=cmap,vmin=-borne,vmax=borne)
                    fig.colorbar(im, ax=ax[i], orientation='vertical')

                    ax[i].set_xlabel('R',fontsize=size_labels)
                    ax[i].set_ylabel('Z',fontsize=size_labels)
                    ax[i].set_aspect('equal','box')
                else:
                    im = ax.tripcolor(R, Z, spectrum_to_plot[0, :], cmap=cmap,vmin=-borne,vmax=borne)
                    fig.colorbar(im, ax=ax, orientation='vertical')
                    ax.set_xlabel('R',fontsize=size_labels)
                    ax.set_ylabel('Z',fontsize=size_labels)
                    ax.set_aspect('equal','box')

            elif should_we_make_separate_plots:
                for coord in coordinates_to_plot:
                    borne = np.max(np.abs(spectrum_to_plot[coord]))
                    if mF != 0:
                        im = ax[coord,i].tripcolor(R, Z, spectrum_to_plot[coord], cmap=cmap,vmin=-borne,vmax=borne)
                        fig.colorbar(im, ax=ax[coord, i], orientation='vertical')

                        ax[coord,i].set_xlabel('R',fontsize=size_labels)
                        ax[coord,i].set_ylabel('Z',fontsize=size_labels)
                        if coord == 0:
                            ax[coord,i].set_title(rf'${field}_r^{axis}$',fontsize=size_labels)
                        elif coord == 1:
                            ax[coord,i].set_title(rf'${field}_{{\theta}}^{axis}$',fontsize=size_labels)
                        elif coord == 2:
                            ax[coord,i].set_title(rf'${field}_z^{axis}$',fontsize=size_labels)
                        ax[coord,i].set_aspect('equal','box')
                    else:
                        im = ax[coord].tripcolor(R, Z, spectrum_to_plot[coord], cmap=cmap,vmin=-borne,vmax=borne)
                        fig.colorbar(im, ax=ax[coord], orientation='vertical')
                        ax[coord].set_xlabel('R',fontsize=size_labels)
                        ax[coord].set_ylabel('Z',fontsize=size_labels)
                        if coord == 0:
                            ax[coord].set_title(rf'${field}_r^{axis}$',fontsize=size_labels)
                        elif coord == 1:
                            ax[coord].set_title(rf'${field}_{{\theta}}^{axis}$',fontsize=size_labels)
                        elif coord == 2:
                            ax[coord].set_title(rf'${field}_z^{axis}$',fontsize=size_labels)
                        ax[coord].set_aspect('equal','box')



            else:
                borne = np.max(np.abs(spectrum_to_plot[1]))
                plt.tripcolor(R, Z, spectrum_to_plot[1], cmap=cmap,vmin=-borne,vmax=borne)

                plt.colorbar()
                plt.quiver(R[::fraction_arrows], Z[::fraction_arrows], spectrum_to_plot[0][::fraction_arrows], spectrum_to_plot[2][::fraction_arrows],
                        scale=length_arrows,
                        scale_units=scale_type,
                        width = width)
#(label='amplitude of space-dependant part')
                plt.xlabel('R',fontsize=size_labels)
                plt.ylabel('Z',fontsize=size_labels)
                #plt.title(f'plot of mode np = {nP}')


    fig.tight_layout()
    if should_we_save_plots:
        if should_we_make_separate_plots:
            plt.savefig(path_out+f'/m{m:03d}/{field}_separate_{def_save_name(m,nP,mF,is_it_from_phys)}')
        else:
            plt.savefig(path_out+f'/m{m:03d}/{field}_gathered_{def_save_name(m,nP,mF,is_it_from_phys)}')
        plt.close()
    else:
        plt.show()


if should_we_draw_mean_field:
    for mF in list_mean_field:
        if should_we_make_separate_plots:
            if mF == 0:
                fig,ax = plt.subplots(3,figsize=(4*factor_plot,12))
            else:
                fig,ax = plt.subplots(3,2,figsize=(8*factor_plot,12))
        else:
            if mF == 0:
                fig,ax = plt.subplots(1,figsize=(4*factor_plot,4))
            else:
                fig,ax = plt.subplots(1,2,figsize=(8*factor_plot,4))
        if mF == 0:
            list_fourier_types = ['c']
        elif mF != 0:
            list_fourier_types = ['c','s']
        amplitude = 0
        for i,axis in enumerate(list_fourier_types):
            if axis != 's' or mF != 0:
                data_file = f'mF{mF}_{axis}.npy'
                print(f'doing mean-field {data_file}')

                # if should_we_plot_opposite:
                #     spectrum_to_plot = -np.load(directory_modes+'/'+data_file)
                # else:
                #     spectrum_to_plot = np.load(directory_modes+'/'+data_file)
                spectrum_to_plot = np.load(path_to_suites+'/mean_field_mesh_sym/'+data_file)
                spectrum_to_plot = rearrange(spectrum_to_plot ," (d N) -> d N",d=D)
                # WEIGHTS = np.array([W for _ in range(D)]).reshape(-1)
                WEIGHTS = np.array([W for _ in range(D)])

                amplitude += np.sqrt(np.sum(WEIGHTS*(spectrum_to_plot**2)))
                spectrum_to_plot /= amplitude

                if should_we_make_separate_plots:
                    for coord in coordinates_to_plot:
                        borne = np.max(np.abs(spectrum_to_plot[coord]))
                        if mF != 0:
                            im = ax[coord,i].tripcolor(R, Z, spectrum_to_plot[coord], cmap=cmap,vmin=-borne,vmax=borne)
                            fig.colorbar(im, ax=ax[coord, i], orientation='vertical')

                            ax[coord,i].set_xlabel('R',fontsize=size_labels)
                            ax[coord,i].set_ylabel('Z',fontsize=size_labels)
                            if coord == 0:
                                ax[coord,i].set_title(rf'${field}_r^{axis}$',fontsize=size_labels)
                            elif coord == 1:
                                ax[coord,i].set_title(rf'${field}_{{\theta}}^{axis}$',fontsize=size_labels)
                            elif coord == 2:
                                ax[coord,i].set_title(rf'${field}_z^{axis}$',fontsize=size_labels)
                            ax[coord,i].set_aspect('equal','box')
                        else:
                            im = ax[coord].tripcolor(R, Z, spectrum_to_plot[coord], cmap=cmap,vmin=-borne,vmax=borne)
                            fig.colorbar(im, ax=ax[coord], orientation='vertical')
                            ax[coord].set_xlabel('R',fontsize=size_labels)
                            ax[coord].set_ylabel('Z',fontsize=size_labels)
                            if coord == 0:
                                ax[coord].set_title(rf'${field}_r^{axis}$',fontsize=size_labels)
                            elif coord == 1:
                                ax[coord].set_title(rf'${field}_{{\theta}}^{axis}$',fontsize=size_labels)
                            elif coord == 2:
                                ax[coord].set_title(rf'${field}_z^{axis}$',fontsize=size_labels)
                            ax[coord].set_aspect('equal','box')



                else:
                    borne = np.max(np.abs(spectrum_to_plot[1]))
                    plt.tripcolor(R, Z, spectrum_to_plot[1], cmap=cmap,vmin=-borne,vmax=borne)

                    plt.colorbar()
                    plt.quiver(R[::fraction_arrows], Z[::fraction_arrows], spectrum_to_plot[0][::fraction_arrows], spectrum_to_plot[2][::fraction_arrows],
                            scale=length_arrows,
                            scale_units=scale_type,
                            width = width)
    #(label='amplitude of space-dependant part')
                    plt.xlabel('R',fontsize=size_labels)
                    plt.ylabel('Z',fontsize=size_labels)
                    #plt.title(f'plot of mode np = {nP}')
        if mF != 0:
            amplitude /= 2
        fig.suptitle(f'Constant amplitude is {amplitude:01F}')
        fig.tight_layout()
        if should_we_save_plots:
            if should_we_make_separate_plots:
                plt.savefig(directory_to_save+f'/mean_{field}_separate_mF{mF}')
            else:
                plt.savefig(directory_to_save+f'/mean_{field}_gathered_mF{mF}')

            plt.close()
        else:
            plt.show()
