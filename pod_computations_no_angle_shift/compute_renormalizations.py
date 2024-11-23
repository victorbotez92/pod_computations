#!/usr/bin/env python3
import sys
import os, array, time
import time
import gc
import struct
from mpi4py import MPI

#from memory_profiler import profile

#sys.path.append("/ccc/cont003/home/limsi/bousquer/einops")
from einops import rearrange
import numpy as np
from scipy.sparse import csr_matrix

###############################
from POD_computation import compute_POD_features,VB_compute_POD_arrays_snaps_method,save_pod,POD  # TO FIX ('VB_compute_POD_arrays_snaps_method' not working -and maybe useless-)
from functions_to_get_data import get_size,get_file,get_data,import_data
from read_restart_sfemans import get_data_from_suites
from read_data import global_parameters
from compute_correlations import build_symmetrized_weights,compute_correlation,core_correlation_matrix,core_correlation_matrix_by_blocks
###############################



def write_job_output(path,message):
    with open(path, 'r') as f:
        content = f.read()
    with open(path, 'w') as f:
        f.write(content+'\n'+message)




# fourier modes to be saved in shape , WARNING : a mode weights a snapshot
########################################################################
########################################################################
########################################################################
########################################################################




#write_job_output(path_to_job_output,f'rank is {rank} out of size {size}')
def renormalization(par,MF,rank,size,mesh_type, comm):
    if par.is_the_field_to_be_renormalized_by_its_L2_norm:
        build_L2_renormalization(par, MF, rank, size, mesh_type, comm)


def build_L2_renormalization(par,MF,rank,size, mesh_type, comm):
    W = np.hstack([np.fromfile(par.path_to_mesh+f"/{mesh_type}weight_S{s:04d}"+par.mesh_ext) for s in range(par.S)]).reshape(-1)
    WEIGHTS = np.array([W for d in range(par.D)]).reshape(-1)
    if "D00" in par.field:
        field_name_in_file = par.field[5:]
    else:
        field_name_in_file = par.field
    # L2_renormalize = [None for _ in range(len(par.paths_to_data))]  # the array that will receive L2 norms for each snapshot
    #output_path = par.path_to_suites + '/' + par.output_path
    for i,path_to_data in enumerate(par.paths_to_data):
        os.makedirs(par.path_to_suites+'/L2_norms/',exist_ok=True)
        if not os.path.exists(par.path_to_suites+'/L2_norms/'+par.output_path+'/'+path_to_data.split('/')[0]+'.npy'): # the calculations are not made if it was already done before
            once_create_array = False
           # new_renormalize_factor = np.zeros(par.snapshots_per_suite)

            for mF in range(rank,MF,size): # beginning calculations
                write_job_output(par.path_to_job_output,f'making step {mF} of path {path_to_data} with rank {rank}')
                print(f'making step {mF} of path {path_to_data}')
                normalize_fourier = (mF == 0) + 1/2*(1-(mF == 0)) # 1/2 if mF > 0, 1 if mF = 0
                if mF == 0:
                    fourier_types = ['c']
                elif mF > 0:
                    fourier_types = ['c','s']
                for a,axis in enumerate(fourier_types):
                ### ==============================================================
                ### importing data
                ### ==============================================================
                    new_data = import_data(par,mF,axis,path_to_data,field_name_in_file) # shape t a (d n)

                    write_job_output(par.path_to_job_output,f"In Compute L2 norm => Import completed for mF={mF}")
                    
                    renormalize_factor = new_data**2*normalize_fourier
                    renormalize_factor = np.sum(WEIGHTS*renormalize_factor,axis=(1,2))

                    if once_create_array == False:
                        new_renormalize_factor = np.copy(renormalize_factor)
                        once_create_array = True
                    else:
                        new_renormalize_factor += renormalize_factor

                    #write_job_output(par.path_to_job_output,f'just finished renormalization for {rank}')

            new_renormalize_factor = np.sqrt(new_renormalize_factor)
            if rank != 0:
                write_job_output(par.path_to_job_output,f'sending the modes from rank = {rank}')
                comm.send(new_renormalize_factor,dest=0,tag=rank)
            elif rank == 0: # gathering all on rank = 0
                for num_rank in range(1,size):
                    write_job_output(par.path_to_job_output,f'trying to receive mode from rank = {num_rank}')
                    new_renormalize_factor += comm.recv(source=num_rank)
                    write_job_output(par.path_to_job_output,f'received the data from from rank = {num_rank}')
                
                print(par.path_to_suites+path_to_data+'L2_norm.npy',new_renormalize_factor)
                np.save(par.path_to_suites+path_to_data+'L2_norm.npy',new_renormalize_factor) # has shape (time)
