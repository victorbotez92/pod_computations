import numpy as np
import matplotlib.pyplot as plt
import os, sys
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

from einops import rearrange

sys.path.append('/gpfs/users/botezv/.venv')
from SFEMaNS_env.read_stb import get_fourier, get_mesh_gauss
from SFEMaNS_env.operators import nodes_to_gauss


path_pod_computations = "/gpfs/users/botezv/APPLICATIONS_POD/pod_computations/codes/"
field = 'u'
#data_file = f"/gpfs/workdir/botezv/MY_APPLICATIONS_SFEMaNS_GIT/LES_VKS_TM73/RUNS/Runs_SFeMANS/VKS_nst/Re500/data_pod_mean_field.txt"
data_file = f"/gpfs/workdir/botezv/MY_APPLICATIONS_SFEMaNS_GIT/LES_VKS_TM73/RUNS/Runs_SFeMANS/VKS_nst/Re500/data_pod_mean_field_penal_div.txt"

parallelize = False
sys.path.append(path_pod_computations)

# from read_data import parameters
from initialization import init

##########################################
##########################################
##########################################

par, sfem_par = init(data_file,parallelize = parallelize)
sfem_par.add_bins(par.complete_output_path+par.output_file_name+f"/phys_pod_modes/m000/",field=f"POD_{par.field}",D=par.D)


##########################################
##########################################
##########################################

field = par.field
path_to_suites = par.complete_output_path+par.output_file_name

path_out = path_to_suites + "/post_pro/"

m_families = [m_family.min() for m_family in par.list_m_families]
##########################################
##########################################
##########################################



def latent_str(m):
    if m != -1:
        return f'a_phys_(mode_time)_m{m}.npy'
    else:
        return f'a_phys_(mode_time).npy'

def symmetry_str(m):
    if m != -1:
        return f'symmetries_phys_m{m}.npy'
    else:
        return f'symmetries_phys.npy'

def spectrum_str(m):
    if m != -1:
        return f'spectrum_phys_m{m}.npy'
    else:
        return f'spectrum_phys.npy'



######################################
######################################
######## Post pro to achieve #########
######################################
######################################

######################################


should_we_plot_pod_spectra = True
nb_pod_lim = 50 #basic value if 15


######################################

should_we_plot_cumulated_energy_percentage = True
should_we_plot_steps = True
percentage_steps = [50,80,95,99,99.9]
percentage_low = None
######################################

should_we_plot_latents_individually = True
list_nP = np.arange(20)
plot_with_symmetrized_individual = False
remove_latents_from_shifts = True
######################################

should_we_plot_latents_collectively = False
list_nPs = [[1,2,3]]
list_nPs_twin = [[0]] # NEEDS TO HAVE SAME LENGTH AS 'list_nPs'
plot_with_symmetrized_collective = False
nb_symmetries = 4#this amounts to the number of angles used for calculations
y_inf_latents_collective = None#-0.35#-0.05#None


apply_change_sign = False
list_signs = [1,1,1,-1,1,1]
if apply_change_sign == False:
    list_signs = np.ones(len(list_nPs[0]))


bbox_to_anchor_collective = (0.75,0.4)#None#
######################################

should_we_plot_latent_vs_latent = False
list_pairs_nP = [((1,0),(2,0))] # first is m, second is nP
plot_with_symmetrized_latent_vs_latent = False

######################################

should_we_plot_fourier_spectra_of_phys_pod_individually = False
list_nP_fourier_of_pod_individual = [[0,0],[0,1],[3,0],[3,1],[3,2],[3,3]]
mF_lim_indiv = None

######################################

should_we_plot_fourier_spectra_of_phys_pod_collectively = True
mF_max = 39
mF_lim_collective = 39
list_modes = np.arange(mF_max+1)
# array_limit = np.concatenate((np.array([0]),np.arange(mF_max//2)*2+1))

# list_lists_nP = [[0,1,2,5]]

# list_lists_nP = [[(0,0),(0,1),(0,2),(0,3),(0,4)],[(0,5),(0,6),(0,7),(0,8),(0,9)],[(0,10),(0,11),(0,12),(0,13),(0,14)]]
# list_lists_nP = [[(0,0),(0,1),(0,2),(0,3),(0,4)],[(1,0),(1,1),(1,2),(1,3),(1,4)],[(2,0),(2,1),(2,2),(2,3),(2,4)],[(3,0),(3,1),(3,2),(3,3),(3,4)],[(4,0),(4,1),(4,2),(4,3),(4,4)]]
list_lists_nP = [[(m, nP) for nP in range(10)] for m in range(5)]# + [[(m, nP) for nP in range(5, 10)] for m in range(1)]
y_inf_fourierspectra_collective = None#5e-8#-0.35#-0.05#

######################################
######################################
######################################
######################################
######################################




os.makedirs(path_out+'latents',exist_ok=True)
os.makedirs(path_out+'fourier_spectra',exist_ok=True)
for m in m_families:
    os.makedirs(path_out+f'latents_{m}',exist_ok=True)
    os.makedirs(path_out+f'fourier_spectra_{m}',exist_ok=True)
path_to_mesh, mesh_ext, mesh_type, S, D, MF = par.path_to_mesh, par.mesh_ext, par.mesh_type, sfem_par.S, par.D, par.MF
_, _, W = get_mesh_gauss(sfem_par)
    

############################
############################
############################     PLOTS OF PHYSICAL SPECTRUM
############################
############################

if should_we_plot_pod_spectra:
    energies_per_family = []
    symmetries_per_family = []
    for m in m_families:
        energies = np.load(path_to_suites + spectrum_str(m))
        if par.should_we_add_mesh_symmetry:
            symmetries = np.load(path_to_suites + symmetry_str(m))
            mesh_symmetries = symmetries.real

        en_nb = np.arange(energies.shape[0])+1
        plt.plot(en_nb, energies,c=colors[0])
        if par.should_we_add_mesh_symmetry:
            indexes_sym = mesh_symmetries>0
            indexes_anti = mesh_symmetries<0
            plt.plot(en_nb[indexes_sym], energies[indexes_sym], 'o',label=f'{par.type_sym}-symmetric POD modes',c=colors[0])
            plt.plot(en_nb[indexes_anti], energies[indexes_anti], '^',label=f'{par.type_sym}-antisymmetric POD modes',c=colors[0])
        plt.semilogy()
        plt.xlim(0,nb_pod_lim)
        plt.ylim(np.abs(energies[nb_pod_lim+1]),None)

        plt.legend(fontsize=12)
        plt.xlabel('POD mode',fontsize=12)
        plt.ylabel('average energy of POD mode',fontsize=12)
        plt.grid(True)
        plt.savefig(path_out+f'physical_POD_spectrum_m{m}')
        plt.close()

    energies = np.load(path_to_suites + spectrum_str(-1))
    if par.should_we_add_mesh_symmetry:
        symmetries = np.load(path_to_suites + symmetry_str(-1))
        mesh_sym = symmetries.real
        m_family = symmetries.imag
    en_nb = np.arange(energies.shape[0])+1
    plt.plot(en_nb, energies, c='black')
    for m in m_families:
        if par.should_we_add_mesh_symmetry:
            mask_sym = np.logical_and(m_family==m, mesh_sym>0)
            mask_anti = np.logical_and(m_family==m, mesh_sym<0)
            plt.plot(en_nb[mask_sym], energies[mask_sym], 'o',c=colors[m],label=f"{m}-family")
            plt.plot(en_nb[mask_anti], energies[mask_anti], '^',c=colors[m])
        else:
            mask = m_family==m
            plt.plot(en_nb[mask], energies[mask], 'D',c=colors[m],label=f"{m}-family")
    plt.semilogy()
    plt.xlim(0,nb_pod_lim)
    plt.ylim(np.abs(energies[nb_pod_lim+1]),None)

    plt.legend(fontsize=12)
    plt.xlabel('POD mode',fontsize=12)
    plt.ylabel('average energy of POD mode',fontsize=12)
    plt.title(f'Global physical POD spectrum')
    plt.grid(True)
    plt.savefig(path_out+f'physical_POD_spectrum')
    plt.close()

############################
############################
############################     PLOTS OF CUMULATED ENERGY
############################
############################

if should_we_plot_cumulated_energy_percentage:
    for m in m_families:
        spectrum = np.load(path_to_suites+spectrum_str(m))
        cumulated_spectrum = np.cumsum(spectrum)
        cumulated_spectrum = cumulated_spectrum/np.max(cumulated_spectrum)*100
        plt.plot(1+np.arange(len(cumulated_spectrum)),cumulated_spectrum)

        if should_we_plot_steps:
            found_all = False
            index = 0
            for i,cumulated_energy in enumerate(cumulated_spectrum):
                if (found_all == False) and (cumulated_energy > percentage_steps[index]):
                    plt.plot([0,i],[cumulated_energy,cumulated_energy],c='black')
                    plt.plot([i,i],[0,cumulated_energy],c='black')
                    index += 1
                    if index == len(percentage_steps):
                        found_all = True
        plt.ylim(percentage_low,None)
        plt.semilogx()
        plt.ylabel('Percentage of energy cumulated')
        plt.xlabel('POD modes')
        plt.title(f'Energy cumulated by POD modes of {m}-family')
        plt.grid(True)
        plt.savefig(path_out+f'physical_cumulated_energy_m{m}')
        plt.close()


    spectrum = np.load(path_to_suites+spectrum_str(-1))
    cumulated_spectrum = np.cumsum(spectrum)
    cumulated_spectrum = cumulated_spectrum/np.max(cumulated_spectrum)*100
    plt.plot(1+np.arange(len(cumulated_spectrum)),cumulated_spectrum)

    if should_we_plot_steps:
        found_all = False
        index = 0
        for i,cumulated_energy in enumerate(cumulated_spectrum):
            if (found_all == False) and (cumulated_energy > percentage_steps[index]):
                plt.plot([0,i],[cumulated_energy,cumulated_energy],c='black')
                plt.plot([i,i],[0,cumulated_energy],c='black')
                index += 1
                if index == len(percentage_steps):
                    found_all = True

    plt.ylim(percentage_low,None)
    plt.semilogx()
    plt.ylabel('Percentage of energy cumulated')
    plt.xlabel('POD modes')
    plt.title(f'Energy cumulated by all POD modes')
    plt.grid(True)
    plt.savefig(path_out+f'physical_cumulated_energy')
    plt.close()

############################
############################
############################     PLOTS OF LATENTS OF POD MODES IN PHYS (individual plots)
############################
############################

if should_we_plot_latents_individually:
    for m in m_families:
        a = np.load(path_to_suites+latent_str(m))
        a = a.T
        
        for nP in list_nP:
            coeff = 1
            if remove_latents_from_shifts:
                coeff *= par.number_shifts
            if not plot_with_symmetrized_individual:
                coeff *= 2*(par.should_we_add_mesh_symmetry) + 1*(1-par.should_we_add_mesh_symmetry)
            plt.plot(a[:a.shape[0]//coeff,nP])
            plt.title(f'mode {nP}')

            plt.ylim(None,None)
            plt.grid(True)
            name_to_save = f'/latents_{m}/phys_nP{nP}'
            plt.savefig(path_out+name_to_save)
            plt.close()

############################
############################
############################     PLOTS OF LATENTS OF POD MODES IN PHYS (collective plots)
############################
############################

if should_we_plot_latents_collectively:

    coeff = 1
    if remove_latents_from_shifts:
        coeff *= par.number_shifts
    if not plot_with_symmetrized_individual:
        coeff *= 2*(par.should_we_add_mesh_symmetry) + 1*(1-par.should_we_add_mesh_symmetry)

    for i,nPs in enumerate(list_nPs):
        fig, ax1 = plt.subplots()
        name_to_save = f'/latents/latent_nPs'
        for j,m,nP in enumerate(nPs):
            a = np.load(path_to_suites+latent_str(m))
            a = a.T
            sign = list_signs[j]
            c = colors[j]
            ax1.plot(sign*a[:a.shape[0]//coeff,nP],label = f'm={m}; n={nP}',c=colors[nP])
            name_to_save += f'_m{m}_nP{nP}'
        plt.grid(True)

        if len(list_nPs_twin[i]) != 0:
            ax2 = ax1.twinx()
            name_to_save += '_twin'
            for m,nP in list_nPs_twin[i]:
                a = np.load(path_to_suites+latent_str(m))
                a = a.T
                ax1.plot(sign*a[:a.shape[0]//coeff,nP],label = f'm={m}; n={nP}',c=colors[nP])
                # ax2.plot(adjusted_time_array[:len(a)//2//nb_symmetries],adjusted_latent[:len(a)//2//nb_symmetries],label = f'n = {nP}',c=colors[nP])
                name_to_save += f'_m{m}_nP{nP}'

            ax2.legend(fontsize=12,loc='upper right')
            ax2.tick_params(axis='y', labelcolor=colors[list_nPs_twin[i][0]])
            ax2.spines['right'].set_color(colors[list_nPs_twin[i][0]])
            # ax2.set_ylim()
        
        # plt.ylim(y_inf_latents_collective,None)
        ax1.legend(fontsize=12,bbox_to_anchor=bbox_to_anchor_collective)#,loc='right'
        # plt.label(bbox_to_anchor = [0.5, 0.2])
        ax1.set_xlabel('Time',fontsize=12)
        ax1.set_ylabel('Amplitudes of POD modes',fontsize=12)

        ax1.set_ylim(None,None)
        ax1.axhline(y=0, color='black', linewidth=1.5)  # Thicker black horizontal axis
        plt.savefig(path_out+name_to_save)
        plt.close()

############################
############################
############################     PLOTS LATENT VS LATENT OF POD MODES IN PHYS (individual plots)
############################
############################

if should_we_plot_latent_vs_latent:
    for i,pair_m,pair_nP in enumerate(list_pairs_nP):
        nP_1,nP_2 = pair_nP
        m_1, m_2 = pair_m
        a1 = np.load(path_to_suites+latent_str(m_1))
        a1 = a1.T
        a2 = np.load(path_to_suites+latent_str(m_2))
        a2 = a2.T
        coeff = 1
        if remove_latents_from_shifts:
            coeff *= par.number_shifts
        if not plot_with_symmetrized_individual:
            coeff *= 2*(par.should_we_add_mesh_symmetry) + 1*(1-par.should_we_add_mesh_symmetry)
        plt.plot(a1[:a1.shape[0]//coeff,nP_1],a2[:a2.shape[0]//coeff,nP_2],'.')
        plt.title(f'modes m,nP = {m,nP_1} and {m,nP_2}')
        plt.ylim(None,None)
        plt.xlabel(f'm,nP = {m_1,nP_1}')
        plt.ylabel(f'm,nP = {m_2,nP_2}')
        plt.grid(True)
        name_to_save = f'/latents/phys_m{m_1}nP{nP_1}_vs_m{m_2}nP{nP_2}'
        plt.savefig(path_out+name_to_save)
        plt.close()

############################
############################
############################     PLOTS OF FOURIER SPECTRA OF POD MODES IN PHYS (individual plots)
############################
############################

if should_we_plot_fourier_spectra_of_phys_pod_individually:
    WEIGHTS = np.array([W for _ in range(D)])
    for m,nP in list_nP_fourier_of_pod_individual:
        sfem_par.add_bins(par.complete_output_path+par.output_file_name+f"/phys_pod_modes/m{m:03d}/",field=f"POD_{par.field}",D=par.D,replace=True)
        POD_mode = get_fourier(sfem_par,nP+1,from_gauss=par.read_from_gauss)[:, :, :mF_lim_indiv]
        if par.read_from_gauss == False:
            POD_mode = nodes_to_gauss(POD_mode, par)
        POD_mode = rearrange(POD_mode, "N (a D) MF -> D N a MF", a=2)[:, :, :, :mF_lim_indiv]
        fourier_spectrum = POD_mode**2*(WEIGHTS.reshape(POD_mode.shape[0], POD_mode.shape[1], 1, 1))
        fourier_spectrum = np.sum(fourier_spectrum, axis=(0,1,2))
        mask = fourier_spectrum/fourier_spectrum.max() > 1e-8
        fourier_spectrum[1:] *= 1/2
        energy_renormalization = np.load(path_to_suites + spectrum_str(m))
        fourier_spectrum *= energy_renormalization[nP]

        plt.plot(np.arange(len(fourier_spectrum))[mask],fourier_spectrum[mask],label=f'm,nP = {m,nP}')
        plt.semilogy()
        plt.xlim(-0.5,20)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.xlabel('Fourier modes',fontsize=12)
        plt.ylabel('Amplitude of Fourier mode',fontsize = 12)
        #plt.savefig(path_out+f'/fourier_spectra/fourier_{fourier_type}_spectrum_nP_{nP:03d}')
        plt.savefig(path_out+f'/fourier_spectra/fourier_spectrum_m{m:03d}_nP{nP:03d}')
        plt.close()



############################
############################
############################     PLOTS OF FOURIER SPECTRA OF POD MODES IN PHYS (collective plots)
############################
############################

if should_we_plot_fourier_spectra_of_phys_pod_collectively:
    WEIGHTS = np.array([W for _ in range(D)])
    for list_m_nP in list_lists_nP:
        name_save = path_out+'/fourier_spectra/fourier_spectrum_nPs'
        print('plotting new Fourier spectrum of a POD mode')
        for m, nP in list_m_nP:
            name_save += f'm{m:03d}_nP{nP:03d}'
            sfem_par.add_bins(par.complete_output_path+par.output_file_name+f"/phys_pod_modes/m{m:03d}/",field=f"POD_{par.field}",D=par.D,replace=True)
            POD_mode = get_fourier(sfem_par,nP+1,from_gauss=par.read_from_gauss)[:, :, :mF_lim_collective]
            if par.read_from_gauss == False:
                POD_mode = nodes_to_gauss(POD_mode, par)
            POD_mode = rearrange(POD_mode, "N (a D) MF -> D N a MF", a=2)[:, :, :, :mF_lim_collective]
            fourier_spectrum = POD_mode**2*(WEIGHTS.reshape(POD_mode.shape[0], POD_mode.shape[1], 1, 1))
            fourier_spectrum = np.sum(fourier_spectrum, axis=(0,1,2))
            fourier_spectrum[1:] *= 1/2
            energy_renormalization = np.load(path_to_suites + spectrum_str(m))
            fourier_spectrum *= energy_renormalization[nP]
            mask = fourier_spectrum/fourier_spectrum.max() > 1e-8
            if par.should_we_add_mesh_symmetry:
                symmetry = np.load(path_to_suites + symmetry_str(m))[nP]
                if symmetry > 0:
                    marker = 'o'
                elif symmetry < 0:
                    marker = '^'
            else:
                marker = 'D'

            plt.plot(np.arange(len(fourier_spectrum))[mask],fourier_spectrum[mask],label=f'm,nP = {m,nP}',linestyle='solid',marker=marker)

        plt.semilogy()
        # plt.xlim(-0.5,20)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.xlabel('Fourier modes',fontsize=12)
        plt.ylabel('Amplitude of Fourier mode',fontsize = 12)
        #plt.savefig(path_out+f'/fourier_spectra/fourier_{fourier_type}_spectrum_nP_{nP:03d}')
        plt.savefig(name_save)
        plt.close()

    # for list_nP in list_lists_nP:
    #     name_save = path_out+'/fourier_spectra/fourier_spectrum_nPs'
    #     print('plotting new Fourier spectrum of a POD mode')
    #     for nP in list_nP:
    #         name_save += f'_{nP:03d}'
    #         energy_per_mF = np.zeros(MF)
    #         print(f'adding nP = {nP}')

    #         if is_there_mesh_symmetry:
    #             symmetries = np.load(path_to_suites + '/symmetries_phys.npy')
    #             if symmetries[nP] > 0:
    #                 marker = 'o'
    #             else:
    #                 marker = '^'
    #         else:
    #             marker = 'D'

    #         for mF in range(mF_max):
    #             if mF in list_modes:
    #                 if not os.path.exists(path_to_suites+f'/phys_pod_modes/nP_{nP:03d}_mF_{mF:03d}_c.npy'):
    #                     all_fourier_types = []
    #                 elif mF != 0 :
    #                     all_fourier_types = ['c','s']
    #                 else:
    #                     all_fourier_types = ['c']
    #                 for fourier_type in all_fourier_types:
    #                     mode_pod = np.load(path_to_suites+f'/phys_pod_modes/nP_{nP:03d}_mF_{mF:03d}_{fourier_type}.npy')
    #                     mode_pod = rearrange(mode_pod ," (d N) -> d N",d=D)
    #                     mode_pod *= mode_pod*WEIGHTS
    #                     if mF != 0:
    #                         mode_pod *= 1/2 ## DIFFERENT NORMALIZATION IF MF != 0
    #                     energy_per_mF[mF] += np.sum(mode_pod)                    
    #         energy_per_mF = np.sqrt(energy_per_mF)
    #         if multiply_by_energy:
    #             energy_per_mF *= list_energies[nP]
    #         if remove_very_low_amplitudes:
    #             array_limit = energy_per_mF/np.max(energy_per_mF) > 1e-4
                # argument = np.argmax(energy_per_mF)
                # parity = argument%2
                # array_limit = np.arange(mF_lim//2)*2+parity
            # plt.plot(np.arange(len(energy_per_mF))[array_limit],energy_per_mF[array_limit],label=f'n = {nP}',linestyle='solid',marker=marker)
            # else:
        #     #     plt.plot(np.arange(len(energy_per_mF))[array_limit],energy_per_mF[array_limit],label=f'nP = {nP}')
        # plt.semilogy()
        # plt.xlim(-0.5,mF_lim)
        # plt.ylim(y_inf_fourierspectra_collective,None)
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # #plt.title(f'Fourier spectrum of nP = {nP} for fourier type {fourier_type}')
        # plt.grid(True)
        # plt.legend(fontsize=12)
        # plt.xlabel('Fourier modes',fontsize=12)
        # plt.ylabel('Amplitude of Fourier mode',fontsize = 12)
        # #plt.savefig(path_out+f'/fourier_spectra/fourier_{fourier_type}_spectrum_nP_{nP:03d}')
        # plt.savefig(name_save)
        # plt.close()
