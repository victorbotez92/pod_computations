#!/usr/bin/env python3
import gc
# from mpi4py import MPI

#from memory_profiler import profile

#sys.path.append("/ccc/cont003/home/limsi/bousquer/einops")
from einops import rearrange
import numpy as np

###############################
from functions_to_get_data import import_data
from basic_functions import write_job_output
###############################

def main_extract_modes(par):
    # if is_the_field_to_be_renormalized_by_magnetic_energy:
    #     with open(path_to_suites+'/fort.75','r') as f:
    #         energies = np.transpose(np.loadtxt(f))
    #     nb_DR = int(field[3])
    #     magnetic_energies = energies[20*nb_DR+10,:]

    if par.should_we_save_Fourier_POD and par.should_we_save_all_fourier_pod_modes:
        a_fourier = np.load(par.complete_output_path+par.output_file_name+f'/latents/cos_mF000.npy') # shape n,T
        Nt_F = a_fourier.shape[-1]
        par.fourier_pod_modes_to_save = np.arange(Nt_F)
        
    for num_mF,mF in enumerate(par.list_modes[par.rank_fourier::par.nb_proc_in_fourier]):
    # for mF in range(par.rank_fourier,par.MF,par.nb_proc_in_fourier):
        if par.should_we_save_phys_POD:
            bool_found = False
            index_phys_mode = 0
            while not bool_found:
                if mF in par.list_m_families[index_phys_mode]:
                    bool_found = True
                else:
                    index_phys_mode += 1
            fourier_family = par.list_m_families[index_phys_mode].min()
            if fourier_family == 0 or (fourier_family == par.number_shifts//2 and par.number_shifts%2 == 0):
                consider_crossed_correlations = False
            else:
                consider_crossed_correlations = True
                if (fourier_family-mF)%par.number_shifts == 0:
                    epsilon_correlations = 1
                else:
                    epsilon_correlations = -1

            a_phys = np.load(par.complete_output_path+par.output_file_name+f"/a_phys_(mode_time)_m{fourier_family}.npy") # signature n,T
            Nt_P = a_phys.shape[-1]//par.number_shifts
            e_phys = np.square(a_phys).sum(-1)/Nt_P

        if par.rank == 0:
            write_job_output(par.path_to_job_output,f'entering Fourier loop {num_mF//par.nb_proc_in_fourier+1}/{len(par.list_modes[par.rank_fourier::par.nb_proc_in_fourier])//par.nb_proc_in_fourier}')
        list_axis = ["c","s"]
        for a in range(par.rank_axis,2,par.nb_proc_in_axis):
            axis = list_axis[a]
            if par.should_we_save_phys_POD:
                phys_pod_modes = None
            if par.should_we_save_Fourier_POD:
                fourier_pod_modes = None
            if (mF,axis) != (0,"s"):
            # #============== mean field if required
            #     if par.should_we_remove_mean_field:
            #         if par.should_mean_field_computation_include_mesh_sym:
            #             char = 'mesh_sym'
            #         else:
            #             char = 'no_mesh_sym'                    
            #         par.mean_field = np.load(par.path_to_suites+f'/mean_field_{char}/mF{mF}_{axis}.npy')
            # #============== mean field if required
                if axis == 'c':
                    fourier_type = 'cos'
                elif axis == 's':
                    fourier_type = 'sin'
                counter_axis = list_axis[1-a]

                if par.should_we_save_Fourier_POD:
                    a_fourier = np.load(par.complete_output_path+par.output_file_name+f'/latents/{fourier_type}_mF{mF:03d}.npy') # shape n,T
                    a_fourier = a_fourier[par.fourier_pod_modes_to_save] # name badly chose, fourier_pod_modes_to_save is an array containing the labels of the POD modes to save for each individual mF
                    Nt_F = a_fourier.shape[-1]
                    e_fourier = np.square(a_fourier).sum(-1)/Nt_F
                for i,path_to_data in enumerate(par.paths_to_data):
                    if par.rank == 0:
                        write_job_output(par.path_to_job_output,f'  Importing {path_to_data}')
                    new_data = import_data(par,mF,axis,path_to_data,par.field_name_in_file) #shape t (d n) [with a being only shape 1]
                    T_new_data = new_data.shape[0]
                    if par.should_we_save_phys_POD and (phys_pod_modes is None):
                        if not consider_crossed_correlations:
                            phys_pod_modes = np.zeros((a_phys.shape[0], new_data.shape[-1]))
                        else:
                            phys_pod_modes = np.zeros((a_phys.shape[0]//2, new_data.shape[-1]))
                        local_nb_snapshots,previous_nb_snapshots = new_data.shape[0],0
                    if par.should_we_save_Fourier_POD and (fourier_pod_modes is None):
                        fourier_pod_modes = np.zeros((a_fourier.shape[0], new_data.shape[-1]))
                    if par.should_we_save_Fourier_POD:
                        fourier_pod_modes += 1/(Nt_F*e_fourier[:,None]) * a_fourier[:,previous_nb_snapshots:local_nb_snapshots]@new_data
                    if par.should_we_save_phys_POD:
                        if not consider_crossed_correlations:
                            phys_pod_modes += 1/(Nt_P*e_phys[:,None]) * a_phys[:,previous_nb_snapshots:local_nb_snapshots]@new_data
                        else:
                            coeff = 1*(axis=="c") + epsilon_correlations*(axis=='s')
                            phys_pod_modes += coeff/(Nt_P*e_phys[::2,None]) * a_phys[::2,previous_nb_snapshots:local_nb_snapshots]@new_data
                    #============================================ Adding the symmetrized part when asked
                    if par.should_we_add_mesh_symmetry:
                        sym_data = rearrange(new_data,"t (d n) -> t d n",d=par.D)
                        del new_data
                        gc.collect()
                        if par.type_sym == 'Rpi':
                            for d in range(par.D):
                                d_coeff = ((d == 0)-1/2)*2 # this coeff has value -1 when d > 1 (so for components theta and z) and 1 when d = 1 (so for component r)
                                sym_data[:,d,:] = d_coeff*sym_data[:,d,par.tab_pairs]
                            if axis == 's':  # factor -1 when performing rpi-sym on sine components
                                sym_data *= (-1)
                        elif par.type_sym == 'centro':
                            for d in range(par.D):
                                d_coeff = (1/2-(d == 2))*2 # this coeff has value +1 when d = 0,1 (so for compo r & theta) and -1 when d = 2 (so for compo z)
                                sym_data[:,d,:] = d_coeff*sym_data[:,d,par.tab_pairs]
                            sym_data *= (-1)**mF# factor -1 when performing centro-sym on even Fourier modes
                        sym_data = rearrange(sym_data,"t d n  -> t (d n)")
                        if par.should_we_save_Fourier_POD:
                            fourier_pod_modes += 1/(Nt_F*e_fourier[:,None]) * a_fourier[:,Nt_F//2+previous_nb_snapshots:Nt_F//2+local_nb_snapshots]@sym_data
                        if par.should_we_save_phys_POD:
                            if not consider_crossed_correlations:
                                phys_pod_modes += 1/(Nt_P*e_phys[:,None]) * a_phys[:,Nt_P//2+previous_nb_snapshots:Nt_P//2+local_nb_snapshots]@sym_data
                            else:
                                coeff = 1*(axis=="c") + epsilon_correlations*(axis=='s')
                                phys_pod_modes += coeff/(Nt_P*e_phys[::2,None]) * a_phys[::2,Nt_P//2+previous_nb_snapshots:Nt_P//2+local_nb_snapshots]@sym_data
                        del sym_data
                        gc.collect()
                    #================================================ Adding crossed correlations when necessary
                    if par.should_we_save_phys_POD and consider_crossed_correlations:
                        new_data = import_data(par,mF,counter_axis,path_to_data,par.field_name_in_file) #shape t (d n) [with a being only shape 1]
                        coeff = -1*(counter_axis=="c") + epsilon_correlations*(counter_axis=='s')
                        phys_pod_modes += coeff/(Nt_P*e_phys[1::2,None]) * a_phys[1::2,previous_nb_snapshots:local_nb_snapshots]@new_data
                        #===================================================== Adding mesh symmetry in this case as well
                        if par.should_we_add_mesh_symmetry:
                            sym_data = rearrange(new_data,"t (d n) -> t d n",d=par.D)
                            del new_data
                            gc.collect()
                            if par.type_sym == 'Rpi':
                                for d in range(par.D):
                                    d_coeff = ((d == 0)-1/2)*2 # this coeff has value -1 when d > 1 (so for components theta and z) and 1 when d = 1 (so for component r)
                                    sym_data[:,d,:] = d_coeff*sym_data[:,d,par.tab_pairs]
                                if axis == 's':  # factor -1 when performing rpi-sym on sine components
                                    sym_data *= (-1)
                            elif par.type_sym == 'centro':
                                for d in range(par.D):
                                    d_coeff = (1/2-(d == 2))*2 # this coeff has value +1 when d = 0,1 (so for compo r & theta) and -1 when d = 2 (so for compo z)
                                    sym_data[:,d,:] = d_coeff*sym_data[:,d,par.tab_pairs]
                                sym_data *= (-1)**mF# factor -1 when performing centro-sym on even Fourier modes
                            sym_data = rearrange(sym_data,"t d n  -> t (d n)")
                            coeff = -1*(counter_axis=="c") + epsilon_correlations*(counter_axis=='s')
                            phys_pod_modes += coeff/(Nt_P*e_phys[1::2,None]) * a_phys[1::2,Nt_P//2+previous_nb_snapshots:Nt_P//2+local_nb_snapshots]@sym_data
                            del sym_data
                            gc.collect()
                    local_nb_snapshots,previous_nb_snapshots = local_nb_snapshots+T_new_data,local_nb_snapshots
                # end for i,path in enumerate(list_paths)
                if par.should_we_save_Fourier_POD:
                    for m_i in range(par.fourier_pod_modes_to_save.size):
                        nP=par.fourier_pod_modes_to_save[m_i]
                        np.save(par.complete_output_path+par.output_file_name+f"/fourier_pod_modes/mF_{mF:03d}_nP_{nP:03d}_{axis}",fourier_pod_modes[m_i])
                if par.should_we_save_phys_POD:
                    for m_i in range(par.phys_pod_modes_to_save.size):
                        nP=par.phys_pod_modes_to_save[m_i]
                        if not consider_crossed_correlations:
                            np.save(par.complete_output_path+par.output_file_name+f"/phys_pod_modes/m_{fourier_family}_nP_{nP:03d}_mF_{mF:03d}_{axis}",phys_pod_modes[m_i])
                        else:
                            coeff = +1*(axis=='c') - 1*(axis=='s')
                            np.save(par.complete_output_path+par.output_file_name+f"/phys_pod_modes/m_{fourier_family}_nP_{2*nP:03d}_mF_{mF:03d}_{axis}",phys_pod_modes[m_i])
                            np.save(par.complete_output_path+par.output_file_name+f"/phys_pod_modes/m_{fourier_family}_nP_{2*nP+1:03d}_mF_{mF:03d}_{counter_axis}",coeff*phys_pod_modes[m_i])
        # end for axis in [cos,sin]
    # end for mF in MF