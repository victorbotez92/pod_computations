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
from POD_computation import compute_POD_features,VB_compute_POD_arrays_snaps_method,save_pod,POD
from functions_to_get_data import get_size,get_file,get_data,import_data
from read_restart_sfemans import get_data_from_suites
from read_data import global_parameters
###############################

def write_job_output(path,message):
    with open(path, 'r') as f:
        content = f.read()
    with open(path, 'w') as f:
        f.write(content+'\n'+message)

###############################

def build_symmetrized_weights(rows,columns,WEIGHTS,WEIGHTS_with_symmetry,D = 3,axis='c'):
    if axis == 's':
        symmetry_coeff = -1
    elif axis == 'c':
        symmetry_coeff = 1
    if D == 3:
        adapted_rows = np.concatenate((rows,rows+len(rows),rows+2*len(rows)))
        adapted_columns = np.concatenate((columns,columns+len(columns),columns+2*len(columns)))
    else:
        adapted_rows = rows
        adapted_columns = columns

    sym_on_right = symmetry_coeff*csr_matrix((WEIGHTS_with_symmetry[adapted_rows],(adapted_rows,adapted_columns)))
    sym_on_left = symmetry_coeff*csr_matrix((WEIGHTS_with_symmetry[adapted_columns],(adapted_rows,adapted_columns)))
    sym_on_right_and_left = csr_matrix((WEIGHTS[adapted_rows],(adapted_columns,adapted_columns)))
    return sym_on_right,sym_on_left,sym_on_right_and_left

def compute_correlation(matrix,give_weights=False,weights = [],with_itself = True,second_matrix = None):   
    Nx = len(matrix)
    if give_weights == False:
        weights = csr_matrix(np.ones(Nx),np.arange(Nx),np.arange(Nx))/Nx
    if with_itself:
        return (matrix.T@weights)@matrix
    else:
        return (matrix.T@weights)@second_matrix

def core_correlation_matrix(par,mF,axis,field_name_in_file,for_building_symmetrized_weights=(None,None,None,None)):
#(matrix,inner_product_weights = [],should_we_use_sparse_matrices=False,symmetrized_weights=None):

    if par.should_we_combine_symmetric_with_non_symmetric == False:
        _,_,WEIGHTS,_ = for_building_symmetrized_weights
    elif par.should_we_use_sparse_matrices:
        rows,columns,WEIGHTS,WEIGHTS_with_symmetry = for_building_symmetrized_weights
        weight_sym_on_right,weight_sym_on_left,weight_sym_on_right_and_left = build_symmetrized_weights(rows,columns,WEIGHTS,
                                                                                    WEIGHTS_with_symmetry,
                                                                                    D = par.D,axis=axis)
            
        sparse_WEIGHTS = csr_matrix((WEIGHTS,(np.arange(len(WEIGHTS)),np.arange(len(WEIGHTS)))))

    ############### ==============================================================
    ############### importing data
    ############### ==============================================================
 
    for i,path_to_data in enumerate(par.paths_to_data):
        new_data = import_data(par,mF,axis,path_to_data,field_name_in_file) #shape t a (d n)
        write_job_output(par.path_to_job_output,f"In POD on Fourier => Import completed for mF={mF}")
        if par.is_the_field_to_be_renormalized_by_its_L2_norm:
            renormalize_factor = np.load(par.path_to_suites+path_to_data+'L2_norm.npy')
            #new_data /= renormalization_factor[:,np.newaxis]
            renormalize_factor = renormalize_factor[:,np.newaxis]
###################### ============================================================
###################### Applying renormalization when asked
###################### ============================================================


        # if par.is_the_field_to_be_renormalized_by_magnetic_energy: # not working
        #     renormalize_factor = np.vstack([magnetic_energies[:,np.newaxis] for _ in range(len(par.paths_to_data))]) 
            
        # elif par.is_the_field_to_be_renormalized_by_its_L2_norm:
        #     new_renormalize = np.load(par.complete_output_path+f'/L2_norms/{par.snapshots_per_suite}snapshots_{par.field}'+path_to_data.split('/')[0]+'.npy')
        #     renormalize_factor = new_renormalize.copy()
        #     renormalize_factor = renormalize_factor[:,np.newaxis]
        else:
            renormalize_factor = 1

        new_data = new_data[:,0]/renormalize_factor

        if i == 0:
            data = np.copy(new_data)
        else:
            data = np.concatenate((data,new_data),axis = 0)
        del new_data
        gc.collect()

    ############### ==============================================================
    ############### Computing correlations
    ############### ==============================================================
 
    if par.should_we_combine_symmetric_with_non_symmetric:
        if par.should_we_use_sparse_matrices:

            bloc_1_1 = compute_correlation(data.T,give_weights=True,weights = sparse_WEIGHTS)
            bloc_2_2 = compute_correlation(data.T,give_weights=True,weights = weight_sym_on_right_and_left)
            bloc_1_2 = compute_correlation(data.T,give_weights=True,weights = weight_sym_on_right)
            #bloc_2_1 = compute_correlation(data.T,give_weights=True,weights = weight_sym_on_left)

            return bloc_1_1,bloc_1_2,bloc_2_2
            # correlation = compute_correlation_matrix(data.T,inner_product_weights = WEIGHTS,
            #                                         should_we_use_sparse_matrices=should_we_use_sparse_matrices,
            #                                         symmetrized_weights=(sym_on_right,sym_on_left,sym_on_right_and_left))
            # del data,sym_on_right,sym_on_left,sym_on_right_and_left
        elif par.should_we_use_sparse_matrices == False:

            sym_data = np.copy(rearrange(data,"t (d n) -> t d n",d=par.D))

            for d in range(par.D):
                d_coeff = ((d == 0)-1/2)*2 # this coeff has value -1 when d > 1 (so for components theta and z) and 1 when d = 1 (so for component r)
                sym_data[:,d,:] = d_coeff*sym_data[:,d,par.tab_pairs]

            if axis == 's':  # factor -1 when performing rpi-sym on sine components
                sym_data *= (-1)

            sym_data = rearrange(sym_data,"t d n  -> t (d n)")
            #sym_data = apply_rpi_symmetry(data,D,tab_pairs)
            data = np.vstack([data,sym_data])
            del sym_data
            gc.collect()
            # correlation = compute_correlation_matrix(full_data.T,inner_product_weights = WEIGHTS,
            #                                         should_we_use_sparse_matrices=should_we_use_sparse_matrices)


    elif par.should_we_combine_symmetric_with_non_symmetric == False:
#    correlation = compute_correlation_matrix(data.T,inner_product_weights = WEIGHTS)
        unique_bloc = compute_correlation(data.T,give_weights=True,weights = WEIGHTS)

        # if should_we_use_sparse_matrices == False:
        #     Nt = len(matrix.T)
        #     Nx = len(matrix)
        #     if len(inner_product_weights) == 0:
        #         inner_product_weights = np.ones(len(matrix))*1/Nx
        #     inner_product_weights = inner_product_weights[:,np.newaxis]
        #     correlation = (matrix.T*inner_product_weights.T)@matrix*1/Nt
        #     del inner_product_weights

        # elif should_we_use_sparse_matrices == True:
        #     Nt = 2*len(matrix.T)
        #     Nx = len(matrix)
        #     sym_on_right,sym_on_left,sym_on_right_and_left = symmetrized_weights
        #     print('inside sparse part of '+"if")
        #     bloc_one_one = (matrix.T*inner_product_weights.T)@matrix
        #     bloc_one_two = (matrix.T@sym_on_right)@matrix
        #     bloc_two_one = (matrix.T@sym_on_left)@matrix
        #     bloc_two_two = (matrix.T@sym_on_right_and_left)@matrix

        #     correlation = np.block([[bloc_one_one,bloc_one_two],
        #                             [bloc_two_one,bloc_two_two]])*1/Nt

        #     del inner_product_weights,bloc_one_one,bloc_one_two,bloc_two_one,bloc_two_two,sym_on_right,sym_on_left,sym_on_right_and_left,symmetrized_weights
        # return correlation
        return unique_bloc

def core_correlation_matrix_by_blocks(par,mF,axis,field_name_in_file,for_building_symmetrized_weights=(None,None,None,None)):

    if par.should_we_combine_symmetric_with_non_symmetric:
            rows,columns,WEIGHTS,WEIGHTS_with_symmetry = for_building_symmetrized_weights
            weight_sym_on_right,weight_sym_on_left,weight_sym_on_right_and_left = build_symmetrized_weights(rows,columns,WEIGHTS,WEIGHTS_with_symmetry,
                                                                                    D = par.D,axis=axis)
    sparse_WEIGHTS = csr_matrix((WEIGHTS,(np.arange(len(WEIGHTS)),np.arange(len(WEIGHTS)))))



    nb_paths = len(par.paths_to_data)
    if par.should_we_combine_symmetric_with_non_symmetric:
        list_blocs = [[[] for _ in range(2*nb_paths)] for _ in range(2*nb_paths)]
    else:
        list_blocs = [[[] for _ in nb_paths] for _ in nb_paths]
        list_blocs_sym = [[[] for _ in range(nb_paths)] for _ in range(nb_paths)]

    ############### ==============================================================
    ############### importing matrix on left
    ############### ==============================================================

    for i,path_to_data in enumerate(par.paths_to_data):
        first_matrix = import_data(par,mF,axis,path_to_data,field_name_in_file)
        write_job_output(par.path_to_job_output,f"In POD on Fourier => Import completed for mF={mF}")

###################### ============================================================
###################### Applying renormalization when asked
###################### ============================================================

        if par.is_the_field_to_be_renormalized_by_its_L2_norm:
            renormalize_factor = np.load(par.path_to_suites+path_to_data+'L2_norm.npy')
            #new_data /= renormalization_factor[:,np.newaxis]
            renormalize_factor = renormalize_factor[:,np.newaxis]

        # if par.is_the_field_to_be_renormalized_by_magnetic_energy: # not working
        #     renormalize_factor = np.vstack([magnetic_energies[:,np.newaxis] for _ in range(len(par.paths_to_data))]) 
            
        # elif par.is_the_field_to_be_renormalized_by_its_L2_norm:
        #     new_renormalize = np.load(par.complete_output_path+f'/L2_norms/{par.snapshots_per_suite}snapshots_{par.field}'+path_to_data.split('/')[0]+'.npy')
        #     renormalize_factor = new_renormalize.copy()
        #     renormalize_factor = renormalize_factor[:,np.newaxis]
        else:
            renormalize_factor = 1

        first_matrix = first_matrix[:,0]/renormalize_factor


###################### ============================================================
###################### Computing auto-correlation
###################### ============================================================

        if par.should_we_combine_symmetric_with_non_symmetric == True:

            list_blocs[i][i] = compute_correlation(first_matrix.T,give_weights=True,weights = sparse_WEIGHTS)
            list_blocs[i][i+nb_paths] = compute_correlation(first_matrix.T,give_weights=True,weights = weight_sym_on_right)
            list_blocs[i+nb_paths][i+nb_paths] = compute_correlation(first_matrix.T,give_weights=True,weights = weight_sym_on_right_and_left)
            
            list_blocs[i+nb_paths][i] = list_blocs[i][i+nb_paths].T

        elif par.should_we_combine_symmetric_with_non_symmetric == False:

            list_blocs[i][i] = compute_correlation(first_matrix.T,give_weights=True,weights = sparse_WEIGHTS)
            list_blocs_sym[i][i] = compute_correlation(first_matrix.T,give_weights=True,weights = weight_sym_on_right_and_left)


    ############### ==============================================================
    ############### importing matrix on right
    ############### ==============================================================

        for j in range(i+1,nb_paths):
            second_matrix = import_data(par,mF,axis,par.paths_to_data[j],field_name_in_file)
            write_job_output(par.path_to_job_output,f"In POD on Fourier => Import completed for mF={mF}")

    ###################### ============================================================
    ###################### Applying renormalization when asked
    ###################### ============================================================

            if par.is_the_field_to_be_renormalized_by_its_L2_norm:
                renormalize_factor = np.load(par.path_to_suites+par.paths_to_data[j]+'L2_norm.npy')
                #new_data /= renormalization_factor[:,np.newaxis]
                renormalize_factor = renormalize_factor[:,np.newaxis]

            # if par.is_the_field_to_be_renormalized_by_magnetic_energy: # not working
            #     renormalize_factor = np.vstack([magnetic_energies[:,np.newaxis] for _ in range(len(par.paths_to_data))]) 
                
            # elif par.is_the_field_to_be_renormalized_by_its_L2_norm:
            #     new_renormalize = np.load(par.complete_output_path+f'/L2_norms/{par.snapshots_per_suite}snapshots_{par.field}'+path_to_data.split('/')[0]+'.npy')
            #     renormalize_factor = new_renormalize.copy()
            #     renormalize_factor = renormalize_factor[:,np.newaxis]
            else:
                renormalize_factor = 1

            second_matrix = second_matrix[:,0]/renormalize_factor

    ############### ==============================================================
    ############### Computing crossed-correlations
    ############### ==============================================================

            if par.should_we_combine_symmetric_with_non_symmetric == True:

                list_blocs[i][j] = compute_correlation(first_matrix.T,give_weights=True,weights = sparse_WEIGHTS,with_itself = False,second_matrix = second_matrix.T)
                list_blocs[i][j+nb_paths] = compute_correlation(first_matrix.T,give_weights=True,weights = weight_sym_on_right,with_itself = False,second_matrix = second_matrix.T)
                list_blocs[i+nb_paths][j] = compute_correlation(first_matrix.T,give_weights=True,weights = weight_sym_on_left,with_itself = False,second_matrix = second_matrix.T)
                list_blocs[i+nb_paths][j+nb_paths] = compute_correlation(first_matrix.T,give_weights=True,weights = weight_sym_on_right_and_left,with_itself = False,second_matrix = second_matrix.T)
                
                list_blocs[j][i] = list_blocs[i][j].T
                list_blocs[j+nb_paths][i] = list_blocs[i][j+nb_paths].T
                list_blocs[j][i+nb_paths] = list_blocs[i+nb_paths][j].T
                list_blocs[j+nb_paths][i+nb_paths] = list_blocs[i+nb_paths][j+nb_paths].T

            elif par.should_we_combine_symmetric_with_non_symmetric == False:

                list_blocs[i][j] = compute_correlation(first_matrix.T,give_weights=True,weights = sparse_WEIGHTS,with_itself = False,second_matrix = second_matrix.T)
                list_blocs_sym[i][j] = compute_correlation(first_matrix.T,give_weights=True,weights = weight_sym_on_right_and_left,with_itself = False,second_matrix = second_matrix.T)

                list_blocs[j][i] = list_blocs[i][j].T
                list_blocs_sym[j][i] = list_blocs_sym[i][j].T
        # end for j in [i+1,nb_paths]
    # end for i,path in paths
    if par.should_we_combine_symmetric_with_non_symmetric == True:
        return list_blocs
    elif par.should_we_combine_symmetric_with_non_symmetric == False:
        return list_blocs,list_blocs_sym