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
from compute_renormalizations import renormalization,build_L2_renormalization
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
########################################################################
##################### CREATE DATA ######################################
########################################################################
########################################################################

data_file = sys.argv[1]
#'data_example.txt'



list_ints = ['D','S','MF','snapshots_per_suite']
list_several_ints = []
list_floats = []
list_bools = ['READ_FROM_SUITE','should_we_combine_symmetric_with_non_symmetric','should_we_use_sparse_matrices',
              'is_the_field_to_be_renormalized_by_magnetic_energy','is_the_field_to_be_renormalized_by_its_L2_norm',
              'should_we_save_Fourier_POD','should_we_save_phys_POD','should_we_save_phys_correlation','should_we_compute_correlation_by_blocks']
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


should_we_save_Fourier_POD = params.should_we_save_Fourier_POD
should_we_save_phys_POD = params.should_we_save_phys_POD

#compute_pod_modes = params.compute_pod_modes
should_we_combine_symmetric_with_non_symmetric = params.should_we_combine_symmetric_with_non_symmetric
should_we_use_sparse_matrices = params.should_we_use_sparse_matrices
should_we_compute_correlation_by_blocks = params.should_we_compute_correlation_by_blocks
should_we_save_phys_correlation = params.should_we_save_phys_correlation

field = params.field

is_the_field_to_be_renormalized_by_magnetic_energy = params.is_the_field_to_be_renormalized_by_magnetic_energy
is_the_field_to_be_renormalized_by_its_L2_norm = params.is_the_field_to_be_renormalized_by_its_L2_norm
renormalize = (is_the_field_to_be_renormalized_by_magnetic_energy or is_the_field_to_be_renormalized_by_its_L2_norm)


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

assert (is_the_field_to_be_renormalized_by_its_L2_norm and is_the_field_to_be_renormalized_by_magnetic_energy) == False
assert (field in list_vv_mesh) or (field in list_H_mesh)

########################################################################
########################################################################
########################################################################
########################################################################


complete_output_path = path_to_suites + '/' + output_path
path_to_job_output = directory_codes + '/JobLogs_outputs/' + name_job_output
params.path_to_job_output = path_to_job_output

os.system(f"touch {directory_codes + '/JobLogs_outputs'}")
os.system(f"touch {path_to_job_output}")

print(path_to_job_output)
write_job_output(path_to_job_output,"entering the code")

########################################################################
########################################################################
################# Make sure all folders are created ####################
########################################################################
########################################################################

if "D00" in field:
    field_name_in_file = field[5:]
else:
    field_name_in_file = field


os.makedirs(complete_output_path,exist_ok=True)
os.makedirs(complete_output_path+"/"+output_file_name,exist_ok=True)

if should_we_combine_symmetric_with_non_symmetric == False:

    os.makedirs(complete_output_path+"/"+output_file_name+"/latents_sym" ,exist_ok=True)
    os.makedirs(complete_output_path+"/"+output_file_name+"/latents_not_sym" ,exist_ok=True)

    os.makedirs(complete_output_path+"/"+output_file_name+"/energies_not_sym" ,exist_ok=True)
    os.makedirs(complete_output_path+"/"+output_file_name+"/energies_sym" ,exist_ok=True)


elif should_we_combine_symmetric_with_non_symmetric == True:

    os.makedirs(complete_output_path+"/"+output_file_name+"/latents" ,exist_ok=True)

    os.makedirs(complete_output_path+"/"+output_file_name+"/energies" ,exist_ok=True)

os.makedirs(complete_output_path+"/"+output_file_name+"/symmetry" ,exist_ok=True)


########################################################################
########################################################################
################# Build pairs/coordinates/weights ######################
########################################################################
########################################################################

R = np.hstack([np.fromfile(path_to_mesh+f"/{mesh_type}rr_S{s:04d}"+mesh_ext) for s in range(S)]).reshape(-1)
Z = np.hstack([np.fromfile(path_to_mesh+f"/{mesh_type}zz_S{s:04d}"+mesh_ext) for s in range(S)]).reshape(-1)
W = np.hstack([np.fromfile(path_to_mesh+f"/{mesh_type}weight_S{s:04d}"+mesh_ext) for s in range(S)]).reshape(-1)
if (should_we_use_sparse_matrices or should_we_compute_correlation_by_blocks) and D == 3 and should_we_combine_symmetric_with_non_symmetric:
    relative_signes = [1,-1,-1] #cf Rpi-symmetry for the three components
    WEIGHTS_with_symmetry = np.array([relative_signes[d]*W for d in range(D)]).reshape(-1)  
WEIGHTS = np.array([W for d in range(D)]).reshape(-1)  

list_pairs = np.load(directory_pairs+pairs)

tab_pairs = np.empty(2*len(list_pairs),dtype=np.int32)
for elm in list_pairs:
    index,sym_index = elm
    tab_pairs[index] = int(sym_index)
    tab_pairs[sym_index] = int(index)

if should_we_use_sparse_matrices or should_we_compute_correlation_by_blocks:
    rows = []
    columns = []
    for elm in list_pairs:
        index,sym_index = elm
        rows.append(index)
        columns.append(sym_index)
        rows.append(sym_index)
        columns.append(index)
    rows = np.array(rows)
    columns = np.array(columns)

    for_building_symmetrized_weights = (rows,columns,WEIGHTS,WEIGHTS_with_symmetry)

elif should_we_combine_symmetric_with_non_symmetric:
    for_building_symmetrized_weights = (None,None,WEIGHTS,None)
########################################################################
########################################################################
############################### Main ###################################
########################################################################
########################################################################


if __name__ == "__main__": 
   
    write_job_output(path_to_job_output,f'rank is {rank} out of size {size}')

    if rank == 0:
        SYMMETRY=[None for _ in range(MF)]
        ENERGY=[None for _ in range(MF)]
        spectrum_cos=[None for _ in range(MF)]
        spectrum_sin=[None for _ in range(MF)]
    # if is_the_field_to_be_renormalized_by_magnetic_energy:
    #     with open(path_to_suites+'/fort.75','r') as f:
    #         energies = np.transpose(np.loadtxt(f))
    #     nb_DR = int(field[3])
    #     magnetic_energies = energies[20*nb_DR+10,:]

    if renormalize:
        renormalization(params,MF,rank,size,mesh_type,comm)


    once_make_cor_for_phys = True
    for mF in range(rank,MF,size):
        write_job_output(path_to_job_output,f'entering loop for mF = {mF}')
        for a,axis in enumerate(["c","s"]):


            write_job_output(path_to_job_output,f"In POD on Fourier => Importing data for mF={mF}")



##################################################################################
##################################################################################
################# Create correlation matrix ######################################
##################################################################################
##################################################################################
            if should_we_compute_correlation_by_blocks == False:

                list_blocs = core_correlation_matrix(params,mF,axis,field_name_in_file,
                                                    for_building_symmetrized_weights=for_building_symmetrized_weights)
                if should_we_use_sparse_matrices:
                    bloc_1_1,bloc_1_2,bloc_2_2 = list_blocs
                    unormalized_correlation = np.block([[bloc_1_1,bloc_1_2],
                                                        [bloc_1_2.T,bloc_2_2]])
                    Nt = len(unormalized_correlation)
                    correlation = 1/Nt*unormalized_correlation
                else:
                    unormalized_correlation = list_blocs
                    Nt = len(unormalized_correlation)
                    correlation = 1/Nt*unormalized_correlation



            elif should_we_compute_correlation_by_blocks == True:
                if should_we_combine_symmetric_with_non_symmetric:
                    correlation = core_correlation_matrix_by_blocks(params,mF,axis,field_name_in_file,
                                                                for_building_symmetrized_weights=for_building_symmetrized_weights)
                    correlation = np.block(correlation)
                    Nt = len(correlation)
                    correlation = 1/Nt*correlation
     
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################

            if should_we_save_Fourier_POD:
        
                pod_a = compute_POD_features(correlation)
                save_pod(pod_a,mF,axis,D,complete_output_path,output_file_name,is_sym = True,sym_combined=True)
                write_job_output(path_to_job_output,f'succesfully saved spectra for mF = {mF}, a = {axis} symetrized suites')
                del pod_a
                gc.collect()
            
            if should_we_save_phys_POD:
                if once_make_cor_for_phys:
                    once_make_cor_for_phys = False
                    if mF == 0:
                        cumulated_correlation = np.copy(correlation)
                    else:
                        cumulated_correlation = 1/2*np.copy(correlation)
                else:
                    cumulated_correlation += 1/2*correlation
                del correlation
                gc.collect()
        # End for a,axis in ['c','s']
    # End for mF in range(rank,MF,size)
    if should_we_save_phys_POD:
        if rank != 0:
            comm.send(cumulated_correlation,dest=0)
            write_job_output(path_to_job_output,f'just sent data from rank {rank} for phys POD')
        if rank == 0:
            for rank_recv in range(1,size):
                write_job_output(path_to_job_output,f'trying to receive correlations from rank = {rank_recv} for phys POD')
                cumulated_correlation += comm.recv(source=rank_recv)
                write_job_output(path_to_job_output,f'just added data from rank {rank_recv} for phys POD')


            pod_a = compute_POD_features(cumulated_correlation)


            save_pod(pod_a,None,None,D,complete_output_path,output_file_name,is_sym = True,sym_combined=True,is_it_phys_pod=True)
            write_job_output(path_to_job_output,f'succesfully saved spectra for symetrized suites (phys POD)')
            if should_we_save_phys_correlation:
                np.save(complete_output_path+'/'+output_file_name+'/phys_correlation.npy',cumulated_correlation)