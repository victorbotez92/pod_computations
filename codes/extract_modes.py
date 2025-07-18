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
import os,sys
###############################

def main_extract_modes(data_par):
    # if is_the_field_to_be_renormalized_by_magnetic_energy:
    #     with open(path_to_suites+'/fort.75','r') as f:
    #         energies = np.transpose(np.loadtxt(f))
    #     nb_DR = int(field[3])
    #     magnetic_energies = energies[20*nb_DR+10,:]

    if data_par.should_we_save_Fourier_POD and data_par.should_we_save_all_fourier_pod_modes:
        a_fourier = np.load(data_par.complete_output_path+data_par.output_file_name+f'/latents/cos_mF000.npy') # shape n,T
        Nt_F = a_fourier.shape[-1]
        data_par.fourier_pod_modes_to_save = np.arange(Nt_F)
        
    for num_mF,mF in enumerate(data_par.list_modes[data_par.rank_fourier::data_par.nb_proc_in_fourier]):
        if data_par.should_we_save_phys_POD:
            bool_found = False
            index_phys_mode = 0
            while not bool_found:
                if mF in data_par.list_m_families[index_phys_mode]:
                    bool_found = True
                else:
                    index_phys_mode += 1
            fourier_family = data_par.list_m_families[index_phys_mode].min()
            if fourier_family == 0 or (fourier_family == data_par.number_shifts//2 and data_par.number_shifts%2 == 0):
                consider_crossed_correlations = False
            else:
                consider_crossed_correlations = True
                if (fourier_family-mF)%data_par.number_shifts == 0:
                    epsilon_correlations = 1
                else:
                    epsilon_correlations = -1
            if consider_crossed_correlations:
                bool_valid_nPs = data_par.phys_pod_modes_to_save.size%2==0
                if not bool_valid_nPs:
                    raise Exception(ValueError, "Code considers crossed correlations: make sure the list of POD modes is of even size")

                for i in range(0, data_par.phys_pod_modes_to_save.size, 2):
                    if data_par.phys_pod_modes_to_save[i] + 1 != data_par.phys_pod_modes_to_save[i+1]:
                        bool_valid_nPs = False
                    if not bool_valid_nPs:
                        raise Exception(ValueError, f"Code considers crossed correlations: make sure for all even nP mode the odd nP+1 is also present (here you required {data_par.phys_pod_modes_to_save[i]}, make sure to require {data_par.phys_pod_modes_to_save[i]+1} as well)")
            a_phys = np.load(data_par.complete_output_path+data_par.output_file_name+f"/a_phys_(mode_time)_m{fourier_family}.npy")[data_par.phys_pod_modes_to_save, :] # signature n,T
            Nt_P = a_phys.shape[-1]//data_par.number_shifts

            e_phys = np.load(data_par.complete_output_path+data_par.output_file_name+f"/spectrum_phys_m{fourier_family}.npy")[data_par.phys_pod_modes_to_save]
            a_phys = a_phys[:, :Nt_P]

            if consider_crossed_correlations:
                dummy_a = a_phys[::2, :] + 1.j*a_phys[1::2, :]
                a_phys = dummy_a
                e_phys = e_phys[::2]

        if data_par.rank == 0:
            write_job_output(data_par.path_to_job_output,f'entering Fourier loop {num_mF//data_par.nb_proc_in_fourier+1}/{len(data_par.list_modes[data_par.rank_fourier::data_par.nb_proc_in_fourier])//data_par.nb_proc_in_fourier}')
        list_axis = ["c","s"]
        for a in range(data_par.rank_axis,2,data_par.nb_proc_in_axis):
            axis = list_axis[a]
            if data_par.should_we_save_phys_POD or data_par.should_we_save_Fourier_POD:
                phys_pod_modes = None
                fourier_pod_modes = None
                local_nb_snapshots = None
                previous_nb_snapshots = None  

            if (axis=='s' and (mF == 0 or consider_crossed_correlations)):
                continue

            if axis == 'c':
                fourier_type = 'cos'
            elif axis == 's':
                fourier_type = 'sin'
            counter_axis = list_axis[1-a]

            if data_par.should_we_save_Fourier_POD:
                a_fourier = np.load(data_par.complete_output_path+data_par.output_file_name+f'/latents/{fourier_type}_mF{mF:03d}.npy') # shape n,T
                a_fourier = a_fourier[data_par.fourier_pod_modes_to_save] # name badly chose, fourier_pod_modes_to_save is an array containing the labels of the POD modes to save for each individual mF
                Nt_F = a_fourier.shape[-1]
                e_fourier = np.square(a_fourier).sum(-1)/Nt_F

            for i,path_to_data in enumerate(data_par.paths_to_data):

                sym_data = None
                
                if data_par.rank == 0:
                    write_job_output(data_par.path_to_job_output,f'  Importing all of {path_to_data}')
                for individual_path_to_data in path_to_data:
                    new_data = import_data(data_par,mF,axis,[individual_path_to_data],data_par.field_name_in_file) #shape t (d n)
                    if data_par.rank == 0:
                        write_job_output(data_par.path_to_job_output,f'      Successfully imported {individual_path_to_data}')
                    T_new_data = new_data.shape[0]
                    N_space = new_data.shape[-1]
                    if (local_nb_snapshots is None) and (previous_nb_snapshots is None):
                        local_nb_snapshots,previous_nb_snapshots = T_new_data,0
                    else:
                        local_nb_snapshots,previous_nb_snapshots = local_nb_snapshots+T_new_data,local_nb_snapshots

                    if data_par.should_we_save_Fourier_POD:
                        if fourier_pod_modes is None:
                            fourier_pod_modes = np.zeros((a_fourier.shape[0], N_space))
                        fourier_pod_modes += 1/(Nt_F*e_fourier[:,None]) * a_fourier[:,previous_nb_snapshots:local_nb_snapshots]@new_data
                        #========== ADDING MESH SYMMETRY
                        if data_par.should_we_add_mesh_symmetry:
                            if data_par.type_sym == 'Rpi':
                                sym_tensor = np.array([1, -1, -1])
                            elif data_par.type_sym == 'centro':
                                sym_tensor = np.array([1, 1, -1])

                            new_data = rearrange(new_data, "t (d n) -> t d n", d=data_par.D)
                            sym_data = np.zeros(new_data.shape)
                            sym_coeff = (data_par.type_sym=='centro')*(-1)**mF + (data_par.type_sym=='Rpi')*(-1)**(axis=='s')
                            for d in range(data_par.D):
                                sym_data[:, d, :] = sym_coeff*sym_tensor[d]*new_data[:, d, data_par.tab_pairs]
                            
                            new_data = rearrange(new_data, "t d n -> t (d n)")
                            sym_data = rearrange(sym_data, "t d n -> t (d n)")

                            fourier_pod_modes += 1/(Nt_F*e_fourier[:,None]) * a_fourier[:,previous_nb_snapshots+Nt_F//2:local_nb_snapshots+Nt_F//2]@sym_data

                    if data_par.should_we_save_phys_POD:

                        if phys_pod_modes is None:
                            phys_pod_modes = np.zeros((a_phys.shape[0], N_space))
                        if consider_crossed_correlations:
                            complex_data = new_data + 1.j*epsilon_correlations*import_data(data_par,mF,"s",[individual_path_to_data],data_par.field_name_in_file)
                            new_data = complex_data
                            del complex_data
                            gc.collect()

                        phys_pod_modes = phys_pod_modes + einsum(1/(Nt_P*data_par.number_shifts*e_phys[:,None]) * a_phys[:,previous_nb_snapshots:local_nb_snapshots], new_data, 'P T, T N -> P N')

                        if data_par.should_we_add_mesh_symmetry:
                            if (consider_crossed_correlations or not data_par.should_we_save_Fourier_POD):
                                if data_par.type_sym == 'Rpi':
                                    sym_tensor = np.array([1, -1, -1])
                                elif data_par.type_sym == 'centro':
                                    sym_tensor = np.array([1, 1, -1])

                                new_data = rearrange(new_data, "t (d n) -> t d n", d=data_par.D)
                                sym_data = np.zeros(new_data.shape)
                                sym_coeff = (data_par.type_sym=='centro')*(-1)**mF + (data_par.type_sym=='Rpi')*(-1)**(axis=='s')
                                for d in range(data_par.D):
                                    if not (consider_crossed_correlations and data_par.type_sym=='Rpi'):
                                        sym_data[:, d, :] = sym_coeff*sym_tensor[d]*new_data[:, d, data_par.tab_pairs]
                                    else:
                                        sym_data = sym_data.astype(np.complex128)
                                        sym_data[:, d, :].real = sym_tensor[d]*new_data[:, d, data_par.tab_pairs].real #no change of sign for cosine (Rpi-sym)
                                        sym_data[:, d, :].imag = -sym_tensor[d]*new_data[:, d, data_par.tab_pairs].imag #change sign for sine (Rpi-sym)
                                new_data = rearrange(new_data, "t d n -> t (d n)")
                                sym_data = rearrange(sym_data, "t d n -> t (d n)")

                            phys_pod_modes = phys_pod_modes + 1/(Nt_P*data_par.number_shifts*e_phys[:,None]) * a_phys[:,previous_nb_snapshots+Nt_P//2:local_nb_snapshots+Nt_P//2]@sym_data

            # end for i,path in enumerate(list_paths)
            _, _, WEIGHTS, _ = data_par.for_building_symmetrized_weights
            if data_par.should_we_save_Fourier_POD:

                for m_i in range(data_par.fourier_pod_modes_to_save.size):
                    nP=data_par.fourier_pod_modes_to_save[m_i]
                    np.save(data_par.complete_output_path+data_par.output_file_name+f"/fourier_pod_modes/mF_{mF:03d}_nP_{nP:03d}_{axis}",fourier_pod_modes[m_i])
            
            if data_par.should_we_save_phys_POD:

                for m_i, nP in enumerate(data_par.phys_pod_modes_to_save):
                    # nP=data_par.phys_pod_modes_to_save[m_i]
                    if not consider_crossed_correlations:
                        np.save(data_par.complete_output_path+data_par.output_file_name+f"/phys_pod_modes/m_{fourier_family}_nP_{nP:03d}_mF_{mF:03d}_{axis}",phys_pod_modes[m_i, :])

                    elif m_i%2==0:
                        to_save = phys_pod_modes[m_i//2, :].real
                        np.save(data_par.complete_output_path+data_par.output_file_name+f"/phys_pod_modes/m_{fourier_family}_nP_{nP:03d}_mF_{mF:03d}_c",to_save)
                        to_save = (1.j*phys_pod_modes[m_i//2, :]).real
                        np.save(data_par.complete_output_path+data_par.output_file_name+f"/phys_pod_modes/m_{fourier_family}_nP_{nP:03d}_mF_{mF:03d}_s",to_save)
                    elif m_i%2 == 1:
                        to_save = phys_pod_modes[(m_i-1)//2, :].imag
                        np.save(data_par.complete_output_path+data_par.output_file_name+f"/phys_pod_modes/m_{fourier_family}_nP_{nP:03d}_mF_{mF:03d}_c",to_save)
                        to_save = (1.j*phys_pod_modes[(m_i-1)//2, :]).imag
                        np.save(data_par.complete_output_path+data_par.output_file_name+f"/phys_pod_modes/m_{fourier_family}_nP_{nP:03d}_mF_{mF:03d}_s",to_save)
                    else:
                        continue
        # end for axis in [cos,sin]
    # end for mF in MF

def switch_to_bins_format(data_par, sfem_par):

    sys.path.append(data_par.path_SFEMaNS_env)
    from read_write_SFEMaNS.write_stb import write_fourier, write_phys
    from read_write_SFEMaNS.read_stb import get_mesh
    from vector_operations.FFT_IFFT import fourier_to_phys
    
    nb_m = len(data_par.list_m_families)
    list_m_nP = [(num_m, nP) for num_m in range(nb_m) for nP in data_par.phys_pod_modes_to_save]
    path_out = data_par.complete_output_path+data_par.output_file_name+f"/phys_pod_modes/"

    _, _, W = get_mesh(sfem_par)

    for i in range(data_par.rank, len(list_m_nP), data_par.size):
        num_m, nP = list_m_nP[i]
        m = data_par.list_m_families[num_m].min()
        pod_fourier_format = np.zeros((len(data_par.R), 2*data_par.D, data_par.list_modes.max()+1))  #(N a*D MF)
        for mF in data_par.list_m_families[num_m]:
            for a, axis in enumerate(["c", "s"]):
                if not(mF == 0 and axis == 's'):
                    new_mode = np.load(data_par.complete_output_path+data_par.output_file_name+f"/phys_pod_modes/m_{m}_nP_{nP:03d}_mF_{mF:03d}_{axis}.npy")
                else:
                    new_mode = np.zeros(data_par.D*data_par.R.shape[0])
                pod_fourier_format[:, a::2, mF] = rearrange(new_mode, '(d n) -> n d', d=data_par.D)
        
        # normalization_factors = np.sum(pod_fourier_format**2*(W.reshape(W.shape[0], 1, 1)), axis=(0,1))
        # normalization_factors[1: ] /= 2
        # normalization_factors = normalization_factors.sum()
        # pod_fourier_format /= normalization_factors
        
        if data_par.bins_format == 'fourier':
            write_fourier(sfem_par,pod_fourier_format,path_out+f"m{m:03d}/",field_name=f'POD_{data_par.field}',I=nP+1)
        elif data_par.bins_format == 'phys':
            pod_phys_format = fourier_to_phys(pod_fourier_format)
            write_phys(sfem_par,pod_phys_format,path_out+f"m{m:03d}/",field_name=f'POD_{data_par.field}',I=nP+1)

    if data_par.size > 1:
        data_par.comm.Barrier()
    for i in range(data_par.rank, len(list_m_nP), data_par.size):
        num_m, nP = list_m_nP[i]
        m = data_par.list_m_families[num_m].min()
        for mF in data_par.list_m_families[num_m]:
            for a, axis in enumerate(["c", "s"]):
                to_remove = f"{data_par.complete_output_path+data_par.output_file_name}/phys_pod_modes/m_{m}_nP_{nP:03d}_mF_{mF:03d}_{axis}.npy"
                if os.path.exists(to_remove):
                    os.remove(to_remove)
    