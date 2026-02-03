#!/usr/bin/env python3
import gc
# from mpi4py import MPI

#from memory_profiler import profile

#sys.path.append("/ccc/cont003/home/limsi/bousquer/einops")
from einops import rearrange, einsum
import numpy as np

###############################
from functions_to_get_data import import_data
from basic_functions import write_job_output
from compute_renormalizations import apply_mesh_sym, coeff_sym_axis
#from POD_computation import apply_mesh_sym, coeff_sym_axis
import os,sys


sys.path.append('/gpfs/users/botezv/.venv')
from SFEMaNS_env.write_stb import write_fourier, write_phys
from SFEMaNS_env.FFT_operations import fourier_to_phys
#from SFEMaNS_env.operators import gauss_to_nodes#, nodes_to_gauss

###############################

def main_extract_modes(inputs):
    # if is_the_field_to_be_renormalized_by_magnetic_energy:
    #     with open(path_to_suites+'/fort.75','r') as f:
    #         energies = np.transpose(np.loadtxt(f))
    #     nb_DR = int(field[3])
    #     magnetic_energies = energies[20*nb_DR+10,:]

    if inputs.should_we_save_Fourier_POD and inputs.should_we_save_all_fourier_pod_modes:
        a_fourier = np.load(inputs.complete_output_path+inputs.output_file_name+f'/latents/cos_mF000.npy') # shape n,T
        Nt_F = a_fourier.shape[-1]
        inputs.fourier_pod_modes_to_save = np.arange(Nt_F)
    consider_crossed_correlations = True #important if nothing to be done   
    for num_mF,mF in enumerate(inputs.list_modes[inputs.rank_fourier::inputs.nb_proc_in_fourier]):
        if inputs.should_we_save_phys_POD:
            bool_found = False
            index_phys_mode = 0
            while not bool_found:
                if mF in inputs.list_m_families[index_phys_mode]:
                    bool_found = True
                else:
                    index_phys_mode += 1
            fourier_family = inputs.list_m_families[index_phys_mode].min()%inputs.number_shifts
            if fourier_family == 0 or (fourier_family == inputs.number_shifts//2 and inputs.number_shifts%2 == 0):
                consider_crossed_correlations = False
            else:
                consider_crossed_correlations = True
                if (fourier_family-mF)%inputs.number_shifts == 0:
                    epsilon_correlations = 1
                else:
                    epsilon_correlations = -1

            if consider_crossed_correlations:
                bool_valid_nPs = inputs.phys_pod_modes_to_save.size%2==0
                if not bool_valid_nPs:
                    raise Exception(ValueError, "Code considers crossed correlations: make sure the list of POD modes is of even size")

                for i in range(0, inputs.phys_pod_modes_to_save.size, 2):
                    if inputs.phys_pod_modes_to_save[i] + 1 != inputs.phys_pod_modes_to_save[i+1]:
                        bool_valid_nPs = False
                    if not bool_valid_nPs:
                        raise Exception(ValueError, f"Code considers crossed correlations: make sure for all even nP mode the odd nP+1 is also present (here you required {inputs.phys_pod_modes_to_save[i]}, make sure to require {inputs.phys_pod_modes_to_save[i]+1} as well)")
            a_phys = np.load(inputs.complete_output_path+inputs.output_file_name+f"/a_phys_m{fourier_family}.npy")[inputs.phys_pod_modes_to_save, :] # signature n,T
            Nt_P = a_phys.shape[-1]//inputs.number_shifts

            e_phys = np.load(inputs.complete_output_path+inputs.output_file_name+f"/spectrum_phys_m{fourier_family}.npy")[inputs.phys_pod_modes_to_save]
            a_phys = a_phys[:, :Nt_P]

            if consider_crossed_correlations:
                dummy_a = a_phys[::2, :] - 1.j*epsilon_correlations*a_phys[1::2, :]
                a_phys = dummy_a
                e_phys = e_phys[::2]

        write_job_output(inputs,f'entering Fourier loop {num_mF//inputs.nb_proc_in_fourier+1}/{len(inputs.list_modes[inputs.rank_fourier::inputs.nb_proc_in_fourier])//inputs.nb_proc_in_fourier}')
        list_axis = ["c","s"]
        for a in range(inputs.rank_axis,2,inputs.nb_proc_in_axis):
            axis = list_axis[a]
            if inputs.should_we_save_phys_POD or inputs.should_we_save_Fourier_POD:
                phys_pod_modes = None
                fourier_pod_modes = None
                local_nb_snapshots = None
                previous_nb_snapshots = None  

            if (axis=='s' and (mF == 0 or consider_crossed_correlations)):
                continue

            counter_axis = list_axis[1-a]

            if inputs.should_we_save_Fourier_POD:
                a_fourier = np.load(inputs.complete_output_path+inputs.output_file_name+f'/latents/{axis}_mF{mF:03d}.npy') # shape n,T
                a_fourier = a_fourier[inputs.fourier_pod_modes_to_save] # name badly chose, fourier_pod_modes_to_save is an array containing the labels of the POD modes to save for each individual mF
                Nt_F = a_fourier.shape[-1]
                e_fourier = np.square(a_fourier).sum(-1)/Nt_F

            for i, (path_to_data, list_T) in enumerate(zip(inputs.paths_to_data, inputs.list_T_per_individual_path)):

                sym_data = None
                
                write_job_output(inputs,f'  Importing all of {path_to_data}')
                for individual_path_to_data,individual_list_T in zip(path_to_data, list_T):
                    print("inputs.list_T_per_individual_path in extract_modes: ", inputs.list_T_per_individual_path)
                    print('list_T in extract_modes: ', list_T)
                    print('individual_list_T in extract_modes: ', individual_list_T)

                    #new_data = import_data(inputs,mF,axis,[individual_path_to_data],inputs.field_name_in_file, rm_mean_field=False) #shape t (d n)
                    new_data = import_data(inputs,mF,[individual_path_to_data],list_T=[individual_list_T])[:, a::2, :] #shape n d t
                    # new_data = import_data(inputs,mF,[individual_path_to_data],list_T=[np.asarray(individual_list_T)])[:, a::2, :] #shape n d t
                    # new_data = nodes_to_gauss(new_data, inputs)
                    write_job_output(inputs,f'      In {i} ==> Successfully imported {individual_path_to_data}')
                    T_new_data = new_data.shape[-1]
                    N_space = new_data.shape[0]
                    if (local_nb_snapshots is None) and (previous_nb_snapshots is None):
                        local_nb_snapshots,previous_nb_snapshots = T_new_data,0
                    else:
                        local_nb_snapshots,previous_nb_snapshots = local_nb_snapshots+T_new_data,local_nb_snapshots

                    #========== ADDING MESH SYMMETRY
                    if inputs.should_we_add_mesh_symmetry:
                        sym_data = np.copy(new_data)
                        apply_mesh_sym(inputs, sym_data, inputs.tab_pairs_nodes, axis, mF)

                    if inputs.should_we_save_Fourier_POD:
                        if fourier_pod_modes is None:
                            fourier_pod_modes = np.zeros((N_space, inputs.D, a_fourier.shape[0]))
                        fourier_pod_modes += 1/Nt_F * einsum(1/e_fourier, a_fourier[:,previous_nb_snapshots:local_nb_snapshots], new_data, "NP, NP T, n D T -> n D NP")

                        #========== ADDING MESH SYMMETRY
                        if inputs.should_we_add_mesh_symmetry:

                            fourier_pod_modes += 1/Nt_F * einsum(1/e_fourier, a_fourier[:,previous_nb_snapshots+Nt_F//2:local_nb_snapshots+Nt_F//2], sym_data, "NP, NP T, n D T -> n D NP")

                    if inputs.should_we_save_phys_POD:

                        if phys_pod_modes is None:
                            if consider_crossed_correlations:
                                phys_pod_modes = np.zeros((N_space, inputs.D, a_phys.shape[0]), dtype=np.complex128)
                            else:
                                phys_pod_modes = np.zeros((N_space, inputs.D, a_phys.shape[0]))

                        if consider_crossed_correlations:
                            complex_data = new_data + 1.j*import_data(inputs,mF,[individual_path_to_data],list_T=[individual_list_T])[:, 1::2, :]
                            # complex_data = new_data + 1.j*import_data(inputs,mF,[individual_path_to_data],list_T=[np.asarray(individual_list_T)])[:, 1::2, :]
                            # complex_data = new_data + 1.j*nodes_to_gauss(import_data(inputs,mF,[individual_path_to_data])[:, 1::2, :], inputs)
                            new_data = complex_data
                            del complex_data
                            gc.collect()

                        phys_pod_modes += 1/Nt_P*einsum(1/e_phys[:], a_phys[:,previous_nb_snapshots:local_nb_snapshots], new_data, 'NP, NP T, n D T -> n D NP')

                        if inputs.should_we_add_mesh_symmetry:
                            if consider_crossed_correlations:
                                
                                sym_data = np.copy(new_data)
                                apply_mesh_sym(inputs, sym_data, inputs.tab_pairs_nodes, axis, mF)
                                if inputs.type_sym == 'Rpi': #different symmetrization for sine and cosine
                                    sym_data = np.conjugate(sym_data)

                            phys_pod_modes += 1/Nt_P*einsum(1/e_phys[:], a_phys[:,previous_nb_snapshots+Nt_P//2:local_nb_snapshots+Nt_P//2], sym_data, 'NP, NP T, n D T -> n D NP')

            # end for i,path in enumerate(list_paths)
            
            if inputs.should_we_save_Fourier_POD:
                # fourier_pod_modes = gauss_to_nodes(fourier_pod_modes, inputs)
                for m_i in range(inputs.fourier_pod_modes_to_save.size):
                    nP=inputs.fourier_pod_modes_to_save[m_i]
                    np.save(inputs.complete_output_path+inputs.output_file_name+f"/fourier_pod_modes/mF_{mF:03d}_nP_{nP:03d}_{axis}",fourier_pod_modes[:, :, m_i])
            
            if inputs.should_we_save_phys_POD:
                if consider_crossed_correlations:
                    dummy1 = np.real(phys_pod_modes)
                    dummy2 = np.imag(phys_pod_modes)
                    # dummy1 = gauss_to_nodes(np.real(phys_pod_modes), inputs)
                    # dummy2 = gauss_to_nodes(np.imag(phys_pod_modes), inputs)
                    phys_pod_modes = dummy1 + 1.j*dummy2
                    del dummy1, dummy2
                    gc.collect()
                # else:
                    # phys_pod_modes = gauss_to_nodes(phys_pod_modes, inputs)

                for m_i, nP in enumerate(inputs.phys_pod_modes_to_save):
    # artificially choose to add factor 1/sqrt(2) to deal with ||u + i tilde(u)||^2 = 2||u||^2
                    factor = 1/2
                    # factor = 1/np.sqrt(2)
                    if not consider_crossed_correlations:
                        np.save(inputs.complete_output_path+inputs.output_file_name+f"/phys_pod_modes/m_{fourier_family}_nP_{nP:03d}_mF_{mF:03d}_{axis}",phys_pod_modes[:, :, m_i])

                    elif m_i%2==0:
                        to_save = factor*phys_pod_modes[:, :, m_i//2].real
                        np.save(inputs.complete_output_path+inputs.output_file_name+f"/phys_pod_modes/m_{fourier_family}_nP_{nP:03d}_mF_{mF:03d}_c",to_save)
                        to_save = factor*phys_pod_modes[:, :, m_i//2].imag
                        np.save(inputs.complete_output_path+inputs.output_file_name+f"/phys_pod_modes/m_{fourier_family}_nP_{nP:03d}_mF_{mF:03d}_s",to_save)
                    elif m_i%2 == 1:
                        to_save = -epsilon_correlations*factor*phys_pod_modes[:, :, (m_i-1)//2].imag
                        np.save(inputs.complete_output_path+inputs.output_file_name+f"/phys_pod_modes/m_{fourier_family}_nP_{nP:03d}_mF_{mF:03d}_c",to_save)
                        to_save = factor*epsilon_correlations*phys_pod_modes[:, :, (m_i-1)//2].real
                        np.save(inputs.complete_output_path+inputs.output_file_name+f"/phys_pod_modes/m_{fourier_family}_nP_{nP:03d}_mF_{mF:03d}_s",to_save)
                    else:
                        continue
        # end for axis in [cos,sin]
    # end for mF in MF

def switch_to_bins_format(inputs, sfem_par):

    nb_m = len(inputs.list_m_families)
    list_m_nP = [(num_m, nP) for num_m in range(nb_m) for nP in inputs.phys_pod_modes_to_save]
    path_out = inputs.complete_output_path+inputs.output_file_name+f"/phys_pod_modes/"

    W = inputs.W
    if inputs.should_we_save_phys_POD:

        for i in range(inputs.rank, len(list_m_nP), inputs.size):
            num_m, nP = list_m_nP[i]
            m = inputs.list_m_families[num_m].min()%inputs.number_shifts
            pod_fourier_format = np.zeros((inputs.R.shape[0], 2*inputs.D, inputs.list_modes.max()+1))  #(N a*D MF)
            for mF in inputs.list_m_families[num_m]:
                for a, axis in enumerate(["c", "s"]):
                    if not(mF == 0 and axis == 's'):
                        new_mode = np.load(inputs.complete_output_path+inputs.output_file_name+f"/phys_pod_modes/m_{m}_nP_{nP:03d}_mF_{mF:03d}_{axis}.npy")
                    else:
                        new_mode = np.zeros((inputs.R.shape[0], inputs.D))
                    pod_fourier_format[:, a::2, mF] = np.copy(new_mode)
                    
                    # pod_fourier_format[:, a::2, mF] = rearrange(new_mode, '(d n) -> n d', d=inputs.D)
            
            # normalization_factors = np.sum(pod_fourier_format**2*(W.reshape(W.shape[0], 1, 1)), axis=(0,1))
            # normalization_factors[1: ] /= 2
            # normalization_factors = normalization_factors.sum()
            # pod_fourier_format /= normalization_factors


            # pod_fourier_format = gauss_to_nodes(pod_fourier_format, inputs)


            if inputs.bins_format == 'fourier':
                write_fourier(sfem_par,pod_fourier_format,path_out+f"m{m:03d}/",field_name=f'POD_{inputs.field}',I=nP+1,from_gauss=False)
            elif inputs.bins_format == 'phys':
                pod_phys_format = fourier_to_phys(pod_fourier_format)
                write_phys(sfem_par,pod_phys_format,path_out+f"m{m:03d}/",field_name=f'POD_{inputs.field}',I=nP+1)
    
        if inputs.size > 1:
            inputs.comm.Barrier()
        for i in range(inputs.rank, len(list_m_nP), inputs.size):
            num_m, nP = list_m_nP[i]
            m = inputs.list_m_families[num_m].min()%inputs.number_shifts
            for mF in inputs.list_m_families[num_m]:
                for a, axis in enumerate(["c", "s"]):
                    to_remove = f"{inputs.complete_output_path+inputs.output_file_name}/phys_pod_modes/m_{m}_nP_{nP:03d}_mF_{mF:03d}_{axis}.npy"
                    if os.path.exists(to_remove):
                        os.remove(to_remove)
       
