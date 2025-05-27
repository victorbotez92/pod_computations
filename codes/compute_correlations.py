#!/usr/bin/env python3

#from memory_profiler import profile

#sys.path.append("/ccc/cont003/home/limsi/bousquer/einops")
import numpy as np
from scipy.sparse import csr_matrix


###############################
from functions_to_get_data import import_data
from basic_functions import write_job_output
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

def compute_correlation(matrix,give_weights=False,weights = [],with_itself = True,second_matrix = None,type_float=np.float64):   
    Nx = len(matrix)
    if give_weights == False:
        weights = csr_matrix(np.ones(Nx),np.arange(Nx),np.arange(Nx))/Nx
    weights = weights.astype(type_float)
    if with_itself:
        return (matrix.T@weights)@matrix
    else:
        return (matrix.T@weights)@second_matrix


def core_correlation_matrix_by_blocks(par,mF,axis,field_name_in_file,for_building_symmetrized_weights=(None,None,None,None),consider_crossed_correlations=False):

    rows,columns,WEIGHTS,WEIGHTS_with_symmetry = for_building_symmetrized_weights

    sparse_WEIGHTS = csr_matrix((WEIGHTS,(np.arange(len(WEIGHTS)),np.arange(len(WEIGHTS))))) # Building a sparse diagonal matrix (required for fct compute_correlation)


    nb_paths = len(par.paths_to_data)
    factor = 1
    if par.should_we_add_mesh_symmetry:
        factor *= 2
        weight_sym_on_right,weight_sym_on_left,weight_sym_on_right_and_left = build_symmetrized_weights(par.type_sym,rows,columns,WEIGHTS,WEIGHTS_with_symmetry,
        D = par.D,axis=axis,mF=mF)
    # No need because already taken into account in nb_paths (Cf path_to_data containing some ".shifted" in initialization.py)
    # if should_we_combine_with_shifted_data: 
    #     factor *= 2
    list_blocs = [[[] for _ in range(factor*nb_paths)] for _ in range(factor*nb_paths)]
    if consider_crossed_correlations:
        list_blocs_crossed = [[[] for _ in range(factor*nb_paths)] for _ in range(factor*nb_paths)]
    else:
        list_blocs_crossed = None
    ############### ==============================================================
    ############### importing matrix on left
    ############### ==============================================================

    for i,path_to_data in enumerate(par.paths_to_data):
        first_matrix = import_data(par,mF,axis,path_to_data,field_name_in_file)
# FIXING BUG ON MEAN-FIELD
        # if par.should_we_remove_mean_field:
        #     first_matrix -= par.mean_field
# FIXING BUG ON MEAN-FIELD
        if par.rank == 0:
            write_job_output(par.path_to_job_output,f"      In POD on Fourier => {path_to_data} imported as left matrix")

###################### ============================================================
###################### Computing auto-correlation
###################### ============================================================

        if par.should_we_add_mesh_symmetry == True:

            list_blocs[i][i] = compute_correlation(first_matrix.T,give_weights=True,weights = sparse_WEIGHTS,type_float=par.type_float)
            list_blocs[i][i+factor//2*nb_paths] = compute_correlation(first_matrix.T,give_weights=True,weights = weight_sym_on_right,type_float=par.type_float)
            list_blocs[i+nb_paths][i+factor//2*nb_paths] = compute_correlation(first_matrix.T,give_weights=True,weights = weight_sym_on_right_and_left,type_float=par.type_float)
            
            list_blocs[i+nb_paths][i] = list_blocs[i][i+factor//2*nb_paths].T

        elif par.should_we_add_mesh_symmetry == False:

            list_blocs[i][i] = compute_correlation(first_matrix.T,give_weights=True,weights = sparse_WEIGHTS,type_float=par.type_float)

    ############### ==============================================================
    ############### importing matrix on right
    ############### ==============================================================

        for j in range(i+1,nb_paths):
            second_matrix = import_data(par,mF,axis,par.paths_to_data[j],field_name_in_file) #shape t a (d n)
# FIXING BUG ON MEAN-FIELD
            # if par.should_we_remove_mean_field:
            #     second_matrix -= par.mean_field
# FIXING BUG ON MEAN-FIELD
            if par.rank == 0:
                write_job_output(par.path_to_job_output,f"          In POD on Fourier => {par.paths_to_data[j]} imported as right matrix")

    ############### ==============================================================
    ############### Computing crossed-correlations (between different paths)
    ############### ==============================================================

            if par.should_we_add_mesh_symmetry == True:

                list_blocs[i][j] = compute_correlation(first_matrix.T,give_weights=True,weights = sparse_WEIGHTS,with_itself = False,second_matrix = second_matrix.T,type_float=par.type_float)

                list_blocs[i][j+factor//2*nb_paths] = compute_correlation(first_matrix.T,give_weights=True,weights = weight_sym_on_right,with_itself = False,second_matrix = second_matrix.T,type_float=par.type_float)
                list_blocs[i+factor//2*nb_paths][j] = compute_correlation(first_matrix.T,give_weights=True,weights = weight_sym_on_left,with_itself = False,second_matrix = second_matrix.T,type_float=par.type_float)
                list_blocs[i+factor//2*nb_paths][j+factor//2*nb_paths] = compute_correlation(first_matrix.T,give_weights=True,weights = weight_sym_on_right_and_left,with_itself = False,second_matrix = second_matrix.T,type_float=par.type_float)
                
                list_blocs[j][i] = list_blocs[i][j].T
                list_blocs[j+factor//2*nb_paths][i] = list_blocs[i][j+factor//2*nb_paths].T
                list_blocs[j][i+factor//2*nb_paths] = list_blocs[i+factor//2*nb_paths][j].T
                list_blocs[j+factor//2*nb_paths][i+factor//2*nb_paths] = list_blocs[i+factor//2*nb_paths][j+factor//2*nb_paths].T

            elif par.should_we_add_mesh_symmetry == False:

                list_blocs[i][j] = compute_correlation(first_matrix.T,give_weights=True,weights = sparse_WEIGHTS,with_itself = False,second_matrix = second_matrix.T,type_float=par.type_float)

                list_blocs[j][i] = list_blocs[i][j].T
        # end for j in [i+1,nb_paths]
    ############### ==============================================================
    ############### importing matrix on right
    ############### ==============================================================
        if consider_crossed_correlations:
            for j in range(nb_paths):
                second_matrix = import_data(par,mF,"s",par.paths_to_data[j],field_name_in_file) #shape t a (d n)
    # FIXING BUG ON MEAN-FIELD
                # if par.should_we_remove_mean_field:
                #     second_matrix -= par.mean_field
    # FIXING BUG ON MEAN-FIELD
                if par.rank == 0:
                    write_job_output(par.path_to_job_output,f"          In crossed POD on Fourier => {par.paths_to_data[j]} imported as right matrix")

        ############### ==============================================================
        ############### Computing crossed-correlations (between cos and sine)
        ############### ==============================================================

                if par.should_we_add_mesh_symmetry == True:

                    list_blocs_crossed[i][j] = compute_correlation(first_matrix.T,give_weights=True,weights = sparse_WEIGHTS,with_itself = False,second_matrix = second_matrix.T,type_float=par.type_float)

                    list_blocs_crossed[i][j+factor//2*nb_paths] = compute_correlation(first_matrix.T,give_weights=True,weights = weight_sym_on_right,with_itself = False,second_matrix = second_matrix.T,type_float=par.type_float)
                    list_blocs_crossed[i+factor//2*nb_paths][j] = compute_correlation(first_matrix.T,give_weights=True,weights = weight_sym_on_left,with_itself = False,second_matrix = second_matrix.T,type_float=par.type_float)
                    list_blocs_crossed[i+factor//2*nb_paths][j+factor//2*nb_paths] = compute_correlation(first_matrix.T,give_weights=True,weights = weight_sym_on_right_and_left,with_itself = False,second_matrix = second_matrix.T,type_float=par.type_float)

                elif par.should_we_add_mesh_symmetry == False:

                    list_blocs_crossed[i][j] = compute_correlation(first_matrix.T,give_weights=True,weights = sparse_WEIGHTS,with_itself = False,second_matrix = second_matrix.T,type_float=par.type_float)
                
            # end for j in [i,nb_paths]
    # end for i,path in paths
    return list_blocs, list_blocs_crossed