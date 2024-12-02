#!/usr/bin/env python3
# import sys
# import os, array, time
# import time
import gc
# import struct
# from mpi4py import MPI

#from memory_profiler import profile

#sys.path.append("/ccc/cont003/home/limsi/bousquer/einops")
# from einops import rearrange
import numpy as np
# from scipy.sparse import csr_matrix

###############################
from POD_computation import compute_POD_features,save_pod
from compute_correlations import core_correlation_matrix_by_blocks
from basic_functions import write_job_output,invert_rank
###############################




def main_extract_latents(par):   

    once_make_cor_for_phys = True
    list_axis = ["c","s"]

    # if is_the_field_to_be_renormalized_by_magnetic_energy:
    #     with open(path_to_suites+'/fort.75','r') as f:
    #         energies = np.transpose(np.loadtxt(f))
    #     nb_DR = int(field[3])
    #     magnetic_energies = energies[20*nb_DR+10,:]

    for mF in range(par.rank_fourier,par.MF,par.nb_proc_in_fourier):
        if par.rank == 0:
            write_job_output(par.path_to_job_output,f'entering Fourier loop {mF//par.nb_proc_in_fourier+1}/{par.MF//par.nb_proc_in_fourier}')
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
            correlation = 1/Nt*correlation
     
    ############### ==============================================================
    ############### MPI_ALL_REDUCE on meridian planes
    ############### ==============================================================
            correlation = par.comm_meridian.reduce(correlation,root=0)

    ############### ==============================================================
    ############### Compute POD of Fourier components
    ############### ==============================================================

            if par.should_we_save_Fourier_POD and par.rank_meridian == 0:
        
                pod_a = compute_POD_features(correlation)
                save_pod(par,pod_a,is_it_phys_pod=False,mF=mF,fourier_type=axis)
                del pod_a
                gc.collect()
            
            if par.should_we_save_phys_POD:
                if once_make_cor_for_phys:
                    once_make_cor_for_phys = False
                    if mF == 0:
                        cumulated_correlation = np.copy(correlation)
                    else:
                        cumulated_correlation = 1/2*np.copy(correlation)
                else:
                    if mF == 0:
                        cumulated_correlation += correlation
                    else:
                        cumulated_correlation += 1/2*correlation
                del correlation
                gc.collect()
        # End for a,axis in ['c','s']
    # End for mF in range(rank,MF,size)
    if par.should_we_save_phys_POD and par.rank_meridian == 0: #mpi_all_reduce on meridian already done above

    ############### ==============================================================
    ############### Compute POD in physical space
    ############### ==============================================================


            ############### MPI_ALL_REDUCE on axis
        cumulated_correlation = par.comm_axis.reduce(cumulated_correlation,root=0)
        if par.rank == 0:
            write_job_output(par.path_to_job_output,'Successfully reduced all in axis')
        if par.rank_axis == 0:
                ############### MPI_ALL_REDUCE in Fourier
            cumulated_correlation = par.comm_fourier.reduce(cumulated_correlation,root=0)
            if par.rank == 0:
                write_job_output(par.path_to_job_output,'Successfully reduced all in Fourier')
            
            if par.rank_fourier == 0:

                pod_a = compute_POD_features(cumulated_correlation)
                save_pod(par,pod_a)
                write_job_output(par.path_to_job_output,f'succesfully saved spectra for symetrized suites (phys POD)')

######################################## OPTIONAL SAVINGS
                if par.should_we_save_phys_correlation:
                    np.save(par.complete_output_path+'/'+par.output_file_name+'/phys_correlation.npy',cumulated_correlation)