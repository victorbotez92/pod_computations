#!/usr/bin/env python3

#from memory_profiler import profile
import numpy as np
from scipy.sparse import csr_matrix
from einops import rearrange

###############################
from functions_to_get_data import import_data
from basic_functions import write_job_output
###############################


###############################
import sys
sys.path.append("/gpfs/users/botezv/.venv")
from SFEMaNS_env.operators import div, nodes_to_gauss
###############################


###############################

def build_symmetrized_weights(type_sym,rows,columns,WEIGHTS,WEIGHTS_with_symmetry,D = 3,axis='c',mF=0):
    if type_sym == 'Rpi':
        if axis == 's':
            symmetry_coeff = -1
        elif axis == 'c':
            symmetry_coeff = 1
    elif type_sym == 'centro':
        symmetry_coeff = (-1)**mF
    else:
        raise ValueError(f'type_sym must be Rpi or centro, not {type_sym}')
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

def compute_correlation(par, mF, matrix, weights = None,with_itself = True,second_matrix = None):   
    Nx = matrix.shape[0]
    type_float = par.type_float

    if weights is None:
        weights = csr_matrix(np.ones(Nx),np.arange(Nx),np.arange(Nx))/Nx
    weights = weights.astype(type_float)
    output = np.empty((matrix.shape[1], matrix.shape[1]))

    if with_itself:
        rearranged_matrix = rearrange(matrix, "(d n) t -> n d t", d = par.D)
        matrix_gauss = nodes_to_gauss(rearranged_matrix, par)
        del rearranged_matrix
        matrix_gauss = rearrange(matrix_gauss, "n d t -> (d n) t")
        output += (matrix_gauss.T@weights)@matrix_gauss

        if par.should_we_penalize_divergence:
            rearranged_matrix = rearrange(matrix, "(d n) t -> n d t", d = par.D)
            matrix_gauss = div(rearranged_matrix, par, par.W, par.R, one_mode = mF)
            del rearranged_matrix
            matrix_gauss = rearrange(matrix_gauss, "n d t -> t (d n)")
            output += par.penal_div(matrix_gauss.T@weights[:weights.shape[0]//3])@matrix_gauss
        del matrix_gauss
        gc.collect()
 
    else:
        rearranged_matrix = rearrange(matrix, "(d n) t -> n d t", d = par.D)
        matrix_gauss = nodes_to_gauss(matrix, par)
        rearranged_matrix = rearrange(second_matrix, "(d n) t -> n d t", d = par.D)
        second_matrix_gauss = nodes_to_gauss(rearranged_matrix, par)
        del rearranged_matrix
        matrix_gauss = rearrange(matrix_gauss, "n d t -> (d n) t")
        second_matrix_gauss = rearrange(second_matrix_gauss, "n d t -> (d n) t")
        output += (matrix_gauss.T@weights)@second_matrix_gauss

        if par.should_we_penalize_divergence:

            rearranged_matrix = rearrange(matrix, "(d n) t -> n d t", d = par.D)
            matrix_gauss = div(rearranged_matrix, par, par.W, par.R, one_mode = mF)
            del rearranged_matrix
            matrix_gauss = rearrange(matrix_gauss, "n d t -> t (d n)")

            rearranged_matrix = rearrange(matrix, "t (d n) -> n d t", d = par.D)
            second_matrix_gauss = div(rearranged_matrix, par, par.W, par.R, one_mode = mF)
            del rearranged_matrix

            output += par.penal_div*(matrix_gauss.T@weights[:weights.shape[0]//3])@second_matrix_gauss

        del matrix_gauss, second_matrix_gauss
        gc.collect()

    return output

def core_correlation_matrix_by_blocks(par,mF,axis,field_name_in_file,for_building_symmetrized_weights=(None,None,None,None),consider_crossed_correlations=False):

    rows,columns,WEIGHTS,WEIGHTS_with_symmetry = for_building_symmetrized_weights

    sparse_WEIGHTS = csr_matrix((WEIGHTS,(np.arange(len(WEIGHTS)),np.arange(len(WEIGHTS))))) # Building a sparse diagonal matrix (required for fct compute_correlation)


    nb_paths = len(par.paths_to_data)
    factor = 1
    if par.should_we_add_mesh_symmetry:
        factor *= 2
        weight_sym_on_right,weight_sym_on_left,weight_sym_on_right_and_left = build_symmetrized_weights(par.type_sym,rows,columns,WEIGHTS,WEIGHTS_with_symmetry,
        D = par.D,axis=axis,mF=mF)
    list_blocs = [[[] for _ in range(factor*nb_paths)] for _ in range(factor*nb_paths)]
    if consider_crossed_correlations:
        list_blocs_crossed = [[[] for _ in range(factor*nb_paths)] for _ in range(factor*nb_paths)]
    else:
        list_blocs_crossed = None
    ############### ==============================================================
    ############### importing matrix on left
    ############### ==============================================================

    for i,path_to_data in enumerate(par.paths_to_data):
        first_matrix = import_data(par,mF,axis,path_to_data,field_name_in_file, rm_mean_field=True)
        write_job_output(par,f"      In POD on Fourier => {path_to_data} imported as left matrix")

###################### ============================================================
###################### Computing auto-correlation
###################### ============================================================

        if par.should_we_add_mesh_symmetry == True:

            list_blocs[i][i] = compute_correlation(par, mF, first_matrix.T,weights = sparse_WEIGHTS)
            list_blocs[i][i+factor//2*nb_paths] = compute_correlation(par, mF, first_matrix.T,weights = weight_sym_on_right)
            list_blocs[i+nb_paths][i+factor//2*nb_paths] = compute_correlation(par, mF, first_matrix.T,weights = weight_sym_on_right_and_left)
            
            list_blocs[i+nb_paths][i] = list_blocs[i][i+factor//2*nb_paths].T

        elif par.should_we_add_mesh_symmetry == False:

            list_blocs[i][i] = compute_correlation(par, mF, first_matrix.T,weights = sparse_WEIGHTS)

    ############### ==============================================================
    ############### importing matrix on right
    ############### ==============================================================

        for j in range(i+1,nb_paths):
            second_matrix = import_data(par,mF,axis,par.paths_to_data[j],field_name_in_file, rm_mean_field=True) #shape t a (d n)
            write_job_output(par,f"          In POD on Fourier => {par.paths_to_data[j]} imported as right matrix")

    ############### ==============================================================
    ############### Computing crossed-correlations (between different paths)
    ############### ==============================================================

            if par.should_we_add_mesh_symmetry == True:

                coeff = -1*(par.type_sym=='Rpi') + 1*(par.type_sym=='centro')

                list_blocs[i][j] = compute_correlation(par, mF, first_matrix.T,weights = sparse_WEIGHTS,with_itself = False,second_matrix = second_matrix.T)

                list_blocs[i][j+factor//2*nb_paths] = compute_correlation(par, mF, first_matrix.T,weights = weight_sym_on_right,with_itself = False,second_matrix = second_matrix.T)
                list_blocs[i+factor//2*nb_paths][j] = compute_correlation(par, mF, first_matrix.T,weights = weight_sym_on_left,with_itself = False,second_matrix = second_matrix.T)
                list_blocs[i+factor//2*nb_paths][j+factor//2*nb_paths] = compute_correlation(par, mF, first_matrix.T,weights = weight_sym_on_right_and_left,with_itself = False,second_matrix = second_matrix.T)
                
                list_blocs[j][i] = list_blocs[i][j].T
                list_blocs[j+factor//2*nb_paths][i] = list_blocs[i][j+factor//2*nb_paths].T
                list_blocs[j][i+factor//2*nb_paths] = list_blocs[i+factor//2*nb_paths][j].T
                list_blocs[j+factor//2*nb_paths][i+factor//2*nb_paths] = list_blocs[i+factor//2*nb_paths][j+factor//2*nb_paths].T

            elif par.should_we_add_mesh_symmetry == False:

                list_blocs[i][j] = compute_correlation(par, mF, first_matrix.T,weights = sparse_WEIGHTS,with_itself = False,second_matrix = second_matrix.T)

                list_blocs[j][i] = list_blocs[i][j].T
        # end for j in [i+1,nb_paths]
    ############### ==============================================================
    ############### importing matrix on right
    ############### ==============================================================
        if consider_crossed_correlations:
            for j in range(nb_paths):
                second_matrix = import_data(par,mF,"s",par.paths_to_data[j],field_name_in_file, rm_mean_field=True) #shape t a (d n)
                write_job_output(par,f"          In crossed POD on Fourier => {par.paths_to_data[j]} imported as right matrix")

        ############### ==============================================================
        ############### Computing crossed-correlations (between cos and sine)
        ############### ==============================================================

                if par.should_we_add_mesh_symmetry == True:

                    coeff = -1*(par.type_sym=='Rpi') + 1*(par.type_sym=='centro')

                    list_blocs_crossed[i][j] = compute_correlation(par, mF, first_matrix.T,weights = sparse_WEIGHTS,with_itself = False,second_matrix = second_matrix.T)

                    list_blocs_crossed[i][j+factor//2*nb_paths] = compute_correlation(par, mF, first_matrix.T,weights = coeff*weight_sym_on_right,with_itself = False,second_matrix = second_matrix.T)
                    list_blocs_crossed[i+factor//2*nb_paths][j] = compute_correlation(par, mF, first_matrix.T,weights = weight_sym_on_left,with_itself = False,second_matrix = second_matrix.T)
                    list_blocs_crossed[i+factor//2*nb_paths][j+factor//2*nb_paths] = compute_correlation(par, mF, first_matrix.T,weights = coeff*weight_sym_on_right_and_left,with_itself = False,second_matrix = second_matrix.T)

                elif par.should_we_add_mesh_symmetry == False:

                    list_blocs_crossed[i][j] = compute_correlation(par, mF, first_matrix.T,weights = sparse_WEIGHTS,with_itself = False,second_matrix = second_matrix.T)
                
            # end for j in [i,nb_paths]
    # end for i,path in paths
    return list_blocs, list_blocs_crossed
