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
from POD_computation import compute_POD_features,save_pod
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

    # if par.should_we_save_phys_POD:
    #     list_correlations = [None for elm in par.list_m_families]

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

            correlation = core_correlation_matrix_by_blocks(par,mF,axis,par.field_name_in_file,
                                                        for_building_symmetrized_weights=par.for_building_symmetrized_weights)
            correlation = np.block(correlation)
            Nt = len(correlation)
            correlation *= par.type_float(1/Nt)
     
    ############### ==============================================================
    ############### MPI_ALL_REDUCE on meridian planes
    ############### ==============================================================
            if par.size > 1:
                correlation = par.comm_meridian.reduce(correlation,root=0)
                if par.rank == 0:
                    write_job_output(par.path_to_job_output,f'{type(correlation)},{correlation.dtype},Successfully reduced all in Meridian')
    ############### ==============================================================
    ############### Compute POD of Fourier components
    ############### ==============================================================

            
            if par.should_we_save_phys_POD:
                if axis == 's' and mF == 0:
                    correlation *= 0
                if i == 0 and a == par.rank_axis:
                    list_correlations = [np.zeros(correlation.shape) for elm in par.list_m_families]
                # if once_make_cor_for_phys:
                #     once_make_cor_for_phys = False
                #     if mF == 0:
                #         list_correlations[index_correlation] = np.copy(correlation)
                #     else:
                #         list_correlations[index_correlation] = par.type_float(1/2)*np.copy(correlation)
                # else:
                #     if mF == 0:
                #         list_correlations[index_correlation] += correlation
                #     else:
                #         list_correlations[index_correlation] += par.type_float(1/2)*correlation
                if mF == 0:
                    list_correlations[index_correlation] += correlation
                else:
                    list_correlations[index_correlation] += par.type_float(1/2)*correlation

            if axis == 's' and mF == 0:
                del correlation
                gc.collect()
                continue

            if par.should_we_save_Fourier_POD and par.rank_meridian == 0:
        
                pod_a = compute_POD_features(par,correlation)
                save_pod(par,pod_a,is_it_phys_pod=False,mF=mF,fourier_type=axis)
                del pod_a
                del correlation
                gc.collect()

                # if par.rank == 0:
                #     write_job_output(par.path_to_job_output,f'{type(cumulated_correlation)},{cumulated_correlation.dtype}')
        # End for a,axis in ['c','s']
    # End for mF in range(rank,MF,size)
    if par.should_we_save_phys_POD and par.rank_meridian == 0: #mpi_all_reduce on meridian already done above
        # for i,elm in enumerate(list_correlations):
        #     if elm is None:
        #         list_correlations[i] = 0*list_correlations[index_correlation]
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
            
            for i,m_family in enumerate(par.list_m_families):
                m = np.min(m_family)
                # arg_m = np.argmin(np.abs(par.list_modes-m))
                # rank_arg_m = invert_rank(rank_fourier,rank_axis,rank_meridian,par)
                if par.rank_fourier == 0:
                # if m in par.list_modes[par.rank_fourier::par.nb_proc_in_fourier]:
                    # pod_a = compute_POD_features(par,cumulated_correlation)
                    pod_a = compute_POD_features(par,list_correlations[i])
                    save_pod(par,pod_a,family=m)
                    write_job_output(par.path_to_job_output,f'succesfully saved spectra for symetrized suites (phys POD) of family {m}')

    ######################################## OPTIONAL SAVINGS
                    if par.should_we_save_phys_correlation:
                        # np.save(par.complete_output_path+'/'+par.output_file_name+f'/phys_correlation.npy',cumulated_correlation)
                        np.save(par.complete_output_path+'/'+par.output_file_name+f'/phys_correlation_m{m}.npy',list_correlations[i])