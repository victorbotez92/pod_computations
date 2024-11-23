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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def write_job_output(path,message):
    with open(path, 'r') as f:
        content = f.read()
    with open(path, 'w') as f:
        f.write(content+'\n'+message)


########################################################################
##################### CREATE DATA ######################################
########################################################################
########################################################################

#data_file = 'data_example.txt'
data_file = sys.argv[1]


list_ints = ['D','S','MF','snapshots_per_suite']
list_several_ints = ['fourier_pod_modes_to_save','phys_pod_modes_to_save']
list_floats = []
list_bools = ['READ_FROM_SUITE','should_we_combine_symmetric_with_non_symmetric','should_we_use_sparse_matrices',
              'is_the_field_to_be_renormalized_by_magnetic_energy','is_the_field_to_be_renormalized_by_its_L2_norm',
              'should_we_save_phys_POD','should_we_save_Fourier_POD','should_we_compute_correlation_by_blocks']
list_chars = ['mesh_ext','path_to_mesh','directory_pairs','directory_codes','field',
              'path_to_suites','name_job_output','output_path','output_file_name']
list_several_chars = ['paths_to_data']


params = global_parameters(data_file,list_ints,list_several_ints,list_floats,list_bools,list_chars,list_several_chars)

list_vv_mesh = ['u','Tub']
list_H_mesh = ['B','H','Mmu','Dsigma']

########################################################################
########################################################################
########################################################################
########################################################################
READ_FROM_SUITE = params.READ_FROM_SUITE

D = params.D
S = params.S
snapshots_per_suite = params.snapshots_per_suite
MF = params.MF

fourier_pod_modes_to_save = params.fourier_pod_modes_to_save
phys_pod_modes_to_save = params.phys_pod_modes_to_save

#compute_pod_modes = params.compute_pod_modes
should_we_combine_symmetric_with_non_symmetric = params.should_we_combine_symmetric_with_non_symmetric
should_we_use_sparse_matrices = params.should_we_use_sparse_matrices
should_we_compute_correlation_by_blocks = params.should_we_compute_correlation_by_blocks

should_we_save_phys_POD = params.should_we_save_phys_POD
should_we_save_Fourier_POD = params.should_we_save_Fourier_POD
field = params.field
is_the_field_to_be_renormalized_by_magnetic_energy = params.is_the_field_to_be_renormalized_by_magnetic_energy
is_the_field_to_be_renormalized_by_its_L2_norm = params.is_the_field_to_be_renormalized_by_its_L2_norm

mesh_ext = params.mesh_ext
path_to_mesh = params.path_to_mesh
directory_pairs = params.directory_pairs
directory_codes = params.directory_codes

path_to_suites = params.path_to_suites
paths_to_data = params.paths_to_data
output_path = params.output_path
output_file_name = params.output_file_name


if field in list_vv_mesh:
    mesh_type = 'vv'
elif field in list_H_mesh:
    mesh_type = 'H'
pairs=f"list_pairs_{mesh_type}.npy"
params.mesh_type = mesh_type

name_job_output = params.name_job_output



########################################################################
########################################################################
########################################################################
########################################################################


complete_output_path = path_to_suites + '/' + output_path
path_to_job_output = directory_codes + '/JobLogs_outputs/' + name_job_output

os.system(f"touch {directory_codes + '/JobLogs_outputs'}")
os.system(f"touch {path_to_job_output}")

write_job_output(path_to_job_output,"entering the code")

########################################################################
########################################################################
########################################################################
########################################################################
########################################################################
########################################################################
########################################################################
########################################################################


if "D00" in field:
    field_name_in_file = field[5:]
else:
    field_name_in_file = field



assert (is_the_field_to_be_renormalized_by_its_L2_norm and is_the_field_to_be_renormalized_by_magnetic_energy) == False

R = np.hstack([np.fromfile(path_to_mesh+f"/{mesh_type}rr_S{s:04d}"+mesh_ext) for s in range(S)]).reshape(-1)
Z = np.hstack([np.fromfile(path_to_mesh+f"/{mesh_type}zz_S{s:04d}"+mesh_ext) for s in range(S)]).reshape(-1)
W = np.hstack([np.fromfile(path_to_mesh+f"/{mesh_type}weight_S{s:04d}"+mesh_ext) for s in range(S)]).reshape(-1)
WEIGHTS = np.array([W for d in range(D)]).reshape(-1)


list_pairs = np.load(directory_pairs+pairs)

tab_pairs = np.empty(2*len(list_pairs),dtype=np.int32)
for elm in list_pairs:
    index,sym_index = elm
    tab_pairs[index] = int(sym_index)
    tab_pairs[sym_index] = int(index)


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
     
    write_job_output(path_to_job_output,f'rank is {rank} out of size {size}')

    if is_the_field_to_be_renormalized_by_magnetic_energy:
        with open(path_to_suites+'/fort.75','r') as f:
            energies = np.transpose(np.loadtxt(f))
        nb_DR = int(field[3])
        magnetic_energies = energies[20*nb_DR+10,:]


    if should_we_save_phys_POD:
        a_phys = np.load(complete_output_path+output_file_name+"/a_phys_(mode_time).npy") # signature n,T
        a_phys = a_phys[phys_pod_modes_to_save]
        Nt = a_phys.shape[-1]
        e_phys = np.square(a_phys).sum(-1)/Nt

    for mF in range(rank,MF,size):
        write_job_output(path_to_job_output,f'entering loop for mF = {mF}')
        if mF == 0:
            list_fourier_types = ['c']
        else:
            list_fourier_types = ['c','s']
        for a,axis in enumerate(list_fourier_types):
            if axis == 'c':
                fourier_type = 'cos'
            elif axis == 's':
                fourier_type = 'sin'
            if should_we_save_Fourier_POD:
                a_fourier = np.load(complete_output_path+output_file_name+f'/latents/{fourier_type}_mF{mF:03d}.npy') # shape n,T
                a_fourier = a_fourier[fourier_pod_modes_to_save]
                Nt = a_fourier.shape[-1]
                e_fourier = np.square(a_fourier).sum(-1)/Nt
            if should_we_compute_correlation_by_blocks:
                for i,path_to_data in enumerate(paths_to_data):
                    new_data = import_data(params,mF,axis,path_to_data,field_name_in_file) #shape t a (d n) [with a being only shape 1]
                ###### Applying renormalization when asked
                    if is_the_field_to_be_renormalized_by_magnetic_energy: # not working
                        renormalize_factor = np.vstack([magnetic_energies[:,np.newaxis] for _ in range(len(paths_to_data))]) 

                    if is_the_field_to_be_renormalized_by_its_L2_norm:
                        renormalize_factor = np.load(path_to_suites+path_to_data+'L2_norm.npy')
                        #new_data /= renormalization_factor[:,np.newaxis]
                        renormalize_factor = renormalize_factor[:,np.newaxis]
                    # elif is_the_field_to_be_renormalized_by_its_L2_norm:
                    #     new_renormalize = np.load(complete_output_path+f'/L2_norms/{snapshots_per_suite}snapshots_{field}'+path_to_data.split('/')[0]+'.npy')
                    #     renormalize_factor = new_renormalize.copy()
                    #     renormalize_factor = renormalize_factor[:,np.newaxis]
                    else:
                        renormalize_factor = 1
                    new_data = new_data[:,0]/renormalize_factor
                #############################################

                    if i == 0:
                        local_nb_snapshots,previous_nb_snapshots = new_data.shape[0],0
                        if should_we_save_Fourier_POD:
                            fourier_pod_modes = 1/(Nt*e_fourier[:,None]) * a_fourier[:,:local_nb_snapshots]@new_data
                        if should_we_save_phys_POD:
                            phys_pod_modes = 1/(Nt*e_phys[:,None]) * a_phys[:,:local_nb_snapshots]@new_data

                    else:
                        local_nb_snapshots,previous_nb_snapshots = local_nb_snapshots+new_data.shape[0],local_nb_snapshots
                        if should_we_save_Fourier_POD:
                            fourier_pod_modes += 1/(Nt*e_fourier[:,None]) * a_fourier[:,previous_nb_snapshots:local_nb_snapshots]@new_data
                        if should_we_save_phys_POD:
                            phys_pod_modes += 1/(Nt*e_phys[:,None]) * a_phys[:,previous_nb_snapshots:local_nb_snapshots]@new_data
                ############## Adding the symmetrized part when asked
                    if should_we_combine_symmetric_with_non_symmetric:
                        sym_data = rearrange(new_data,"t (d n) -> t d n",d=D)
                        del new_data
                        gc.collect()
                        for d in range(D):
                            d_coeff = ((d == 0)-1/2)*2 # this coeff has value -1 when d > 1 (so for components theta and z) and 1 when d = 1 (so for component r)
                            sym_data[:,d,:] = d_coeff*sym_data[:,d,tab_pairs]

                        if axis == 's':  # factor -1 when performing rpi-sym on sine components
                            sym_data *= (-1)

                        sym_data = rearrange(sym_data,"t d n  -> t (d n)")
                        if should_we_save_Fourier_POD: #debugging
                            # diff = np.abs(np.abs(a_fourier[:,Nt//2+previous_nb_snapshots:Nt//2+local_nb_snapshots])-np.abs(a_fourier[:,previous_nb_snapshots:local_nb_snapshots]))
                            # diff = diff/min(np.max(np.abs(a_fourier[:,Nt//2+previous_nb_snapshots:Nt//2+local_nb_snapshots])),np.max(np.abs(a_fourier[:,previous_nb_snapshots:local_nb_snapshots])))
                            # diff = np.sum(diff)/len(diff)
                            # assert diff < 1e-8
                            # if rank == 0:
                            #     write_job_output(path_to_job_output,f'test passed for mF {mF} axis {axis} path_to_data {path_to_data}')
                            # print(diff,f'test passed for mF {mF} axis {axis} path_to_data {path_to_data}')
                            # a = 0
                            # print(1/a)
                            fourier_pod_modes += 1/(Nt*e_fourier[:,None]) * a_fourier[:,Nt//2+previous_nb_snapshots:Nt//2+local_nb_snapshots]@sym_data
                        if should_we_save_phys_POD:
                            phys_pod_modes += 1/(Nt*e_phys[:,None]) * a_phys[:,Nt//2+previous_nb_snapshots:Nt//2+local_nb_snapshots]@sym_data
                        del sym_data
                        gc.collect()
                #############################################
            # save in npy in fourier space

                    # if i == 0:
                    #     combined_data = np.copy(data)
                    # else:
                    #     combined_data = np.concatenate((combined_data,data),axis = 0)
                    # del data
                    # gc.collect()


                # end for path_to_data in paths_to_data
            # endif should_we_compute_correlation_by_blocks
            elif should_we_compute_correlation_by_blocks == False:
                for i,path_to_data in enumerate(paths_to_data):
                    new_data = import_data(params,mF,axis,path_to_data,field_name_in_file) #shape t a (d n) [with a being only shape 1]
                ###### Applying renormalization when asked
                    if is_the_field_to_be_renormalized_by_magnetic_energy: # not working
                        renormalize_factor = np.vstack([magnetic_energies[:,np.newaxis] for _ in range(len(paths_to_data))]) 
                        
                    if is_the_field_to_be_renormalized_by_its_L2_norm:
                        renormalize_factor = np.load(path_to_suites+path_to_data+'L2_norm.npy')
                        #new_data /= renormalization_factor[:,np.newaxis]
                        renormalize_factor = renormalize_factor[:,np.newaxis]

                    else:
                        renormalize_factor = 1
                    new_data = new_data[:,0]/renormalize_factor

                    if i == 0:
                        combined_data = np.copy(new_data)
                    else:
                        combined_data = np.concatenate((combined_data,new_data),axis = 0)
                    del new_data
                    gc.collect()                        
                ###### Computing symmetric when possible
                if should_we_combine_symmetric_with_non_symmetric:
                    sym_data = rearrange(combined_data,"t (d n) -> t d n",d=D)

                    for d in range(D):
                        d_coeff = ((d == 0)-1/2)*2 # this coeff has value -1 when d > 1 (so for components theta and z) and 1 when d = 1 (so for component r)
                        sym_data[:,d,:] = d_coeff*sym_data[:,d,tab_pairs]

                    if axis == 's':  # factor -1 when performing rpi-sym on sine components
                        sym_data *= (-1)

                    sym_data = rearrange(sym_data,"t d n  -> t (d n)")
                    combined_data = np.r_[combined_data,sym_data]
                    del sym_data
                    gc.collect()
                if should_we_save_Fourier_POD:
                    fourier_pod_modes = 1/(Nt*e_fourier[:,None]) * a_fourier@combined_data
                if should_we_save_phys_POD:
                    phys_pod_modes = 1/(Nt*e_phys[:,None]) * a_phys@combined_data
                del combined_data
                gc.collect()
            # end elif should_we_combine_correlation_by_block == False
            if should_we_save_Fourier_POD:
                os.makedirs(complete_output_path+output_file_name+"/fourier_pod_modes",exist_ok=True)
                for m_i in range(fourier_pod_modes_to_save.size):
                    nP=fourier_pod_modes_to_save[m_i]
                    np.save(complete_output_path+output_file_name+f"/fourier_pod_modes/mF_{mF:03d}_nP_{nP:03d}_{axis}",fourier_pod_modes[m_i])
            if should_we_save_phys_POD:
                os.makedirs(complete_output_path+output_file_name+"/phys_pod_modes",exist_ok=True)
                for m_i in range(phys_pod_modes_to_save.size):
                    nP=phys_pod_modes_to_save[m_i]
                    np.save(complete_output_path+output_file_name+f"/phys_pod_modes/nP_{nP:03d}_mF_{mF:03d}_{axis}",phys_pod_modes[m_i])
        # end for axis in [cos,sin]
    # end for mF in MF
            # if should_we_combine_symmetric_with_non_symmetric:

            #     sym_data = np.copy(rearrange(combined_data,"t (d n) -> t d n",d=D))

            #     for d in range(D):
            #         d_coeff = ((d == 0)-1/2)*2 # this coeff has value -1 when d > 1 (so for components theta and z) and 1 when d = 1 (so for component r)
            #         sym_data[:,d,:] = d_coeff*sym_data[:,d,tab_pairs]

            #     if axis == 's':  # factor -1 when performing rpi-sym on sine components
            #         sym_data *= (-1)

            #     sym_data = rearrange(sym_data,"t d n  -> t (d n)")
                
            #     # concatenation without copy
            #     full_data = np.r_[combined_data,sym_data]
            #     del combined_data,sym_data
                          
            # else :
            #     full_data = combined_data 
            #     del combined_data
            # gc.collect()
            # #computing fourier pod modes
            # a_fourier = np.load(complete_output_path+output_file_name+f"/latents/cos_mF{mF:03d}.npy")
            # a_fourier = a_fourier[fourier_pod_modes_to_save]
            # Nt = a_fourier.shape[-1]
            # e_fourier = np.square(a_fourier).sum(-1)/Nt
            # fourier_pod_modes = 1/(Nt*e_fourier[:,None]) * a_fourier@full_data
            # fourier_pod_modes = rearrange(fourier_pod_modes ," n (d N) -> n d N",d=D)

            # # save in npy in fourier space
            # os.makedirs(complete_output_path+output_file_name+"/fourier_pod_modes",exist_ok=True)
            # for m_i in range(fourier_pod_modes_to_save.size):
            #     nP=phys_pod_modes_to_save[m_i]
            #     np.save(complete_output_path+output_file_name+f"/fourier_pod_modes/mF_{mF:03d}_nP_{nP:03d}_{axis}",fourier_pod_modes[m_i])

            #computing phys pod modes
            # a_phys = np.load(complete_output_path+output_file_name+"/a_phys_(mode_time).npy")
            # a_phys = a_phys[phys_pod_modes_to_save]
            # e_phys = np.square(a_phys).sum(-1)/Nt
            # phys_pod_modes = 1/(Nt*e_phys[:,None]) * a_phys@full_data
            # phys_pod_modes = rearrange(phys_pod_modes ," n (d N) -> n d N",d=D)
            
            # # save in npy in fourier space
            # os.makedirs(complete_output_path+output_file_name+"/phys_pod_modes",exist_ok=True)
            # for m_i in range(phys_pod_modes_to_save.size):
            #     nP=phys_pod_modes_to_save[m_i]
            #     np.save(complete_output_path+output_file_name+f"/phys_pod_modes/nP_{nP:03d}_mF_{mF:03d}_{axis}",phys_pod_modes[m_i])
            











