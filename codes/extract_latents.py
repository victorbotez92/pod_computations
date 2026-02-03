#!/usr/bin/env python3
import gc, os
# from memory_profiler import profile

import numpy as np

###############################
from POD_computation import compute_POD_features, save_pod, POD, update_pod_with_mean_field
from compute_correlations import core_correlation_matrix_by_blocks
from basic_functions import write_job_output, print_memory_usage
###############################



#@profile
def main_extract_latents(inputs):   

    list_axis = ["c","s"]

    # if is_the_field_to_be_renormalized_by_magnetic_energy:
    #     with open(path_to_suites+'/fort.75','r') as f:
    #         energies = np.transpose(np.loadtxt(f))
    #     nb_DR = int(field[3])
    #     magnetic_energies = energies[20*nb_DR+10,:]

    if inputs.should_we_save_phys_POD:
        list_correlations = None
        list_crossed_correlations = None

    for i,mF in enumerate(inputs.list_modes[inputs.rank_fourier::inputs.nb_proc_in_fourier]):
        if inputs.should_we_save_phys_POD:
            bool_found = False
            index_correlation = 0
            while not bool_found:
                if mF in inputs.list_m_families[index_correlation]:
                    bool_found = True
                else:
                    index_correlation += 1

        write_job_output(inputs,f'entering Fourier loop {i//inputs.nb_proc_in_fourier+1}/{len(inputs.list_modes[inputs.rank_fourier::inputs.nb_proc_in_fourier])//inputs.nb_proc_in_fourier}')
        print_memory_usage(inputs, tag="Before entering axis loop:")

        for a in range(inputs.rank_axis,2,inputs.nb_proc_in_axis):
            axis = list_axis[a]

    ############### ==============================================================
    ############### Create correlation matrix
    ############### ==============================================================
            if inputs.should_we_save_phys_POD:
                m = np.min(inputs.list_m_families[index_correlation])%inputs.number_shifts
                if m == 0 or (m == inputs.number_shifts//2 and inputs.number_shifts%2 == 0):
                    consider_crossed_correlations = False
                else:
                    consider_crossed_correlations = True
                    if (m-mF)%inputs.number_shifts == 0:
                        epsilon_correlations = 1
                    elif (m+mF)%inputs.number_shifts == 0:
                        epsilon_correlations = -1
                    else:
                        raise ValueError(f"Inconsistency in crossed correlations: code found mF={mF} in {m}-family")

            else:
                consider_crossed_correlations = False

            print_memory_usage(inputs, tag="Before starting correlation computations:")

            #correlation, crossed_correlations = core_correlation_matrix_by_blocks(inputs,mF,axis,inputs.field_name_in_file,
            all_blocks, all_blocks_crossed = core_correlation_matrix_by_blocks(inputs,mF,axis,consider_crossed_correlations=consider_crossed_correlations)
            
            if inputs.size > 1:
                inputs.comm.Barrier()
                write_job_output(inputs, f"Successfully computed correlation matrices at a={a}, Fourier loop {i//inputs.nb_proc_in_fourier+1}/{len(inputs.list_modes[inputs.rank_fourier::inputs.nb_proc_in_fourier])//inputs.nb_proc_in_fourier}")

            correlation = np.block(all_blocks.list_blocs)
            Nt = len(correlation)
            correlation *= inputs.type_float(1/Nt)
            if consider_crossed_correlations:
                crossed_correlations = np.block(all_blocks_crossed.list_blocs)

                crossed_correlations *= inputs.type_float(1/Nt)


            del all_blocks, all_blocks_crossed
            gc.collect()
    ############### ==============================================================
    ############### MPI_ALL_REDUCE on meridian planes
    ############### ==============================================================
            if inputs.size > 1:
                correlation = inputs.comm_meridian.reduce(correlation,root=0)
                # if inputs.should_we_penalize_divergence: #separated from correlation for POD on cos&sin
                #     dvg_correlation = inputs.comm_meridian.reduce(dvg_correlation,root=0)
                if consider_crossed_correlations: #already contains dvg correlations
                    crossed_correlations = inputs.comm_meridian.reduce(crossed_correlations,root=0)
                write_job_output(inputs,'Successfully reduced all in Meridian')
    ############### ==============================================================
    ############### Compute POD of Fourier components
    ############### ==============================================================

            
            if inputs.should_we_save_phys_POD:
                if axis == 's' and mF == 0:
                    correlation *= 0
                if list_correlations is None:
                # if i == 0 and a == inputs.rank_axis:
                    list_correlations = np.array([np.zeros(correlation.shape) for _ in inputs.list_m_families])
                if list_crossed_correlations is None:
                    list_crossed_correlations = np.array([np.zeros(correlation.shape) for _ in inputs.list_m_families])
                # if inputs.should_we_penalize_divergence:
                #     total_correlation = correlation + dvg_correlation
                # else:
                #     total_correlation = correlation
                total_correlation = np.copy(correlation)
                if axis == 's' and mF == 0:
                    total_correlation *= 0

                if mF == 0:
                    list_correlations[index_correlation] += total_correlation
                else:
                    list_correlations[index_correlation] += inputs.type_float(1/2)*total_correlation
                    if consider_crossed_correlations:
                        list_crossed_correlations[index_correlation] += -inputs.type_float(1/2)*epsilon_correlations*crossed_correlations
                        # list_correlations[index_correlation] += -1.j*inputs.type_float(1/2)*epsilon_correlations*crossed_correlations
                        # np.save(inputs.complete_output_path+'/'+inputs.output_file_name+f'/crossed_correlation_mF{mF}_{axis}.npy',crossed_correlations)
                        # np.save(inputs.complete_output_path+'/'+inputs.output_file_name+f'/crossed_correlation_mF{mF}_{axis}.npy',-1.j*inputs.type_float(1/2)*epsilon_correlations*crossed_correlations)

            if axis == 's' and mF == 0:
                del correlation
                gc.collect()
                continue

            if inputs.should_we_save_Fourier_POD and inputs.rank_meridian == 0:
        
                pod_a = compute_POD_features(inputs,correlation,mF=mF,axis=axis)
                if inputs.should_we_remove_mean_field:
                    pod_a = update_pod_with_mean_field(inputs,pod_a,is_it_phys_pod=False,mF=mF,axis=axis)
                save_pod(inputs,pod_a,is_it_phys_pod=False,mF=mF,axis=axis)

                np.save(inputs.complete_output_path+'/'+inputs.output_file_name+f'/fourier_correlation_mF{mF}_{axis}.npy',correlation)

                del pod_a
                del correlation
                gc.collect()

        # End for a,axis in ['c','s']
    # End for mF in range(rank,MF,size)

    if inputs.should_we_save_phys_POD and inputs.rank_meridian == 0: #mpi_all_reduce on meridian already done above
        # list_correlations = np.asarray(list_correlations)
    ############### ==============================================================
    ############### Compute POD in physical space
    ############### ==============================================================


            ############### MPI_ALL_REDUCE on axis
        if inputs.size > 1:
            # cumulated_correlation = inputs.comm_axis.reduce(cumulated_correlation,root=0)
            list_correlations = inputs.comm_axis.reduce(list_correlations,root=0)
            # if consider_crossed_correlations:
                # list_crossed_correlations = inputs.comm_axis.reduce(list_crossed_correlations,root=0)
            list_crossed_correlations = inputs.comm_axis.reduce(list_crossed_correlations,root=0)

            write_job_output(inputs,'Successfully reduced all in axis')
        if inputs.rank_axis == 0:
                ############### MPI_ALL_REDUCE in Fourier
            if inputs.size > 1:
                # cumulated_correlation = inputs.comm_fourier.reduce(cumulated_correlation,root=0)
                list_correlations = inputs.comm_fourier.reduce(list_correlations,root=0)
                # if consider_crossed_correlations:
                    # list_crossed_correlations = inputs.comm_fourier.reduce(list_crossed_correlations,root=0)
                list_crossed_correlations = inputs.comm_fourier.reduce(list_crossed_correlations,root=0)

                write_job_output(inputs,'Successfully reduced all in Fourier')
            
            list_pod_a = []
            for i,m_family in enumerate(inputs.list_m_families):
                m = np.min(m_family)%inputs.number_shifts
                if inputs.rank_fourier == 0:
                    if m == 0 or (m == inputs.number_shifts//2 and inputs.number_shifts%2 == 0):
                        consider_crossed_correlations = False
                    else:
                        consider_crossed_correlations = True
                        if (m-mF)%inputs.number_shifts == 0:
                            epsilon_correlations = 1
                        else:
                            epsilon_correlations = -1

                    pod_a = compute_POD_features(inputs,list_correlations[i]+1.j*list_crossed_correlations[i],family=m,consider_crossed_correlations=consider_crossed_correlations)

                    list_pod_a.append(pod_a)
                    if inputs.should_we_remove_mean_field:
                        pod_a = update_pod_with_mean_field(inputs,pod_a,family=m)
                    save_pod(inputs,pod_a,family=m)
                    write_job_output(inputs,f'succesfully saved spectra for symetrized suites (phys POD) of family {m}')

                    if inputs.should_we_save_phys_correlation:
                        np.save(inputs.complete_output_path+'/'+inputs.output_file_name+f'/phys_correlation_m{m}.npy',list_correlations[i])
                        if consider_crossed_correlations:
                            np.save(inputs.complete_output_path+'/'+inputs.output_file_name+f'/phys_crossed_correlation_m{m}.npy',list_crossed_correlations[i])

    ######################################## OPTIONAL SAVINGS
                    # if inputs.should_we_save_phys_correlation:
                        # np.save(inputs.complete_output_path+'/'+inputs.output_file_name+f'/phys_correlation_m{m}.npy',list_correlations[i])
            if inputs.rank_fourier == 0:
                all_eigvals = [pod.eigvals for pod in list_pod_a]
                all_eigvals = np.concatenate(all_eigvals)
                sorting_indexes = np.argsort(all_eigvals)[::-1] #sort in decreasing order
                all_eigvals = all_eigvals[sorting_indexes]
                all_eigvecs = np.vstack([pod.proj_coeffs for pod in list_pod_a])[sorting_indexes, :]

                all_symmetries = np.concatenate([pod.symmetries for pod in list_pod_a])
                all_symmetries = all_symmetries[sorting_indexes]
                full_pod = POD(all_eigvals,all_eigvecs,all_symmetries)
                save_pod(inputs,full_pod)
                write_job_output(inputs,f'succesfully saved full spectra for symetrized suites (phys POD)')
