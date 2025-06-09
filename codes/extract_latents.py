#!/usr/bin/env python3
# import sys
# import os, array, time
# import time
import gc, os
# import struct
# from mpi4py import MPI

# from memory_profiler import profile

#sys.path.append("/ccc/cont003/home/limsi/bousquer/einops")
# from einops import rearrange
import numpy as np
# from scipy.sparse import csr_matrix

###############################
from POD_computation import compute_POD_features, save_pod, POD
from compute_correlations import core_correlation_matrix_by_blocks
from basic_functions import write_job_output
###############################



#@profile
def main_extract_latents(par):   

    once_make_cor_for_phys = True
    list_axis = ["c","s"]

    # if is_the_field_to_be_renormalized_by_magnetic_energy:
    #     with open(path_to_suites+'/fort.75','r') as f:
    #         energies = np.transpose(np.loadtxt(f))
    #     nb_DR = int(field[3])
    #     magnetic_energies = energies[20*nb_DR+10,:]

    for i,mF in enumerate(par.list_modes[par.rank_fourier::par.nb_proc_in_fourier]):
        if par.should_we_save_phys_POD:
            bool_found = False
            index_correlation = 0
            while not bool_found:
                if mF in par.list_m_families[index_correlation]:
                    bool_found = True
                else:
                    index_correlation += 1

        if par.rank == 0:
            write_job_output(par.path_to_job_output,f'entering Fourier loop {i//par.nb_proc_in_fourier+1}/{len(par.list_modes[par.rank_fourier::par.nb_proc_in_fourier])//par.nb_proc_in_fourier}')
        for a in range(par.rank_axis,2,par.nb_proc_in_axis):
            axis = list_axis[a]
            if par.rank == 0:
                write_job_output(par.path_to_job_output,f'doing axis {axis}')

    ############### ==============================================================
    ############### Create correlation matrix
    ############### ==============================================================
            if par.should_we_save_phys_POD and axis == 'c':
                m = par.list_m_families[index_correlation][0]
                if m == 0 or (m == par.number_shifts//2 and par.number_shifts%2 == 0):
                    consider_crossed_correlations = False
                else:
                    consider_crossed_correlations = True
                    if (m-mF)%par.number_shifts == 0:
                        epsilon_correlations = 1
                    else:
                        epsilon_correlations = -1

            else:
                consider_crossed_correlations = False

            correlation, crossed_correlations = core_correlation_matrix_by_blocks(par,mF,axis,par.field_name_in_file,
                                                        for_building_symmetrized_weights=par.for_building_symmetrized_weights,
                                                        consider_crossed_correlations=consider_crossed_correlations)
            correlation = np.block(correlation)
            Nt = len(correlation)
            correlation *= par.type_float(1/Nt)
            if consider_crossed_correlations:
                crossed_correlations = np.block(crossed_correlations)
                crossed_correlations *= par.type_float(1/Nt)
                crossed_correlations = 1/2*(crossed_correlations-crossed_correlations.T)
    ############### ==============================================================
    ############### MPI_ALL_REDUCE on meridian planes
    ############### ==============================================================
            if par.size > 1:
                correlation = par.comm_meridian.reduce(correlation,root=0)
                if consider_crossed_correlations:
                    crossed_correlations = par.comm_meridian.reduce(crossed_correlations,root=0)
                if par.rank == 0:
                    write_job_output(par.path_to_job_output,'Successfully reduced all in Meridian')
    ############### ==============================================================
    ############### Compute POD of Fourier components
    ############### ==============================================================

            
            if par.should_we_save_phys_POD:
                if axis == 's' and mF == 0:
                    correlation *= 0
                if i == 0 and a == par.rank_axis:
                    list_correlations = [np.zeros(correlation.shape, dtype=np.complex128) for _ in par.list_m_families]

                if mF == 0:
                    list_correlations[index_correlation] += correlation
                    # if consider_crossed_correlations:
                    #     list_correlations[index_correlation] += 2*1.j*epsilon_correlations*crossed_correlations
                else:
                    list_correlations[index_correlation] += par.type_float(1/2)*correlation
                    if consider_crossed_correlations:
                        list_correlations[index_correlation] += 1.j*epsilon_correlations*crossed_correlations

            if axis == 's' and mF == 0:
                del correlation
                gc.collect()
                continue

            if par.should_we_save_Fourier_POD and par.rank_meridian == 0:
        
                pod_a = compute_POD_features(par,correlation,mF=mF,a=axis)
                save_pod(par,pod_a,is_it_phys_pod=False,mF=mF,fourier_type=axis)
                del pod_a
                del correlation
                gc.collect()

        # End for a,axis in ['c','s']
    # End for mF in range(rank,MF,size)
    if par.should_we_save_phys_POD and par.rank_meridian == 0: #mpi_all_reduce on meridian already done above
        list_correlations = np.array(list_correlations)
    ############### ==============================================================
    ############### Compute POD in physical space
    ############### ==============================================================


            ############### MPI_ALL_REDUCE on axis
        if par.size > 1:
            # cumulated_correlation = par.comm_axis.reduce(cumulated_correlation,root=0)
            list_correlations = par.comm_axis.reduce(list_correlations,root=0)
            if par.rank == 0:
                write_job_output(par.path_to_job_output,'Successfully reduced all in axis')
        if par.rank_axis == 0:
                ############### MPI_ALL_REDUCE in Fourier
            if par.size > 1:
                # cumulated_correlation = par.comm_fourier.reduce(cumulated_correlation,root=0)
                list_correlations = par.comm_fourier.reduce(list_correlations,root=0)
                if par.rank == 0:
                    write_job_output(par.path_to_job_output,'Successfully reduced all in Fourier')
            
            list_pod_a = []
            for i,m_family in enumerate(par.list_m_families):
                m = np.min(m_family)
                if par.rank_fourier == 0:
                    if m == 0 or (m == par.number_shifts//2 and par.number_shifts%2 == 0):
                        consider_crossed_correlations = False
                    else:
                        consider_crossed_correlations = True
                        if (m-mF)%par.number_shifts == 0:
                            epsilon_correlations = 1
                        else:
                            epsilon_correlations = -1

                    pod_a = compute_POD_features(par,1/2*(list_correlations[i]+np.conjugate(list_correlations[i].T)),family=m,consider_crossed_correlations=consider_crossed_correlations)
                    list_pod_a.append(pod_a)
                    # if par.number_shifts > 1:
                    save_pod(par,pod_a,family=m)
                    write_job_output(par.path_to_job_output,f'succesfully saved spectra for symetrized suites (phys POD) of family {m}')

    ######################################## OPTIONAL SAVINGS
                    if par.should_we_save_phys_correlation:
                        # np.save(par.complete_output_path+'/'+par.output_file_name+f'/phys_correlation.npy',cumulated_correlation)
                        np.save(par.complete_output_path+'/'+par.output_file_name+f'/phys_correlation_m{m}.npy',list_correlations[i])
            if par.rank_fourier == 0:
                all_eigvals = [pod.eigvals for pod in list_pod_a]
                all_eigvals = np.concatenate(all_eigvals)
                sorting_indexes = np.argsort(all_eigvals)[::-1] #sort in decreasing order
                all_eigvals = all_eigvals[sorting_indexes]
                all_eigvecs = np.vstack([pod.proj_coeffs for pod in list_pod_a])[sorting_indexes, :]
                all_symmetries = np.concatenate([pod.symmetries for pod in list_pod_a])[sorting_indexes]
                full_pod = POD(all_eigvals,all_eigvecs,all_symmetries)
                save_pod(par,full_pod)
                write_job_output(par.path_to_job_output,f'succesfully saved full spectra for symetrized suites (phys POD)')
