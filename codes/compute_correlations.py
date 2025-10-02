#!/usr/bin/env python3

#from memory_profiler import profile
import gc

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

#def build_symmetrized_weights(type_sym,rows,columns,WEIGHTS,WEIGHTS_with_symmetry,D = 3,axis='c',mF=0):
#    if type_sym == 'Rpi':
#        if axis == 's':
#            symmetry_coeff = -1
#        elif axis == 'c':
#            symmetry_coeff = 1
#    elif type_sym == 'centro':
#        symmetry_coeff = (-1)**mF
#    else:
#        raise ValueError(f'type_sym must be Rpi or centro, not {type_sym}')
#    if D == 3:
#        adapted_rows = np.concatenate((rows,rows+len(rows),rows+2*len(rows)))
#        adapted_columns = np.concatenate((columns,columns+len(columns),columns+2*len(columns)))
#    else:
#        adapted_rows = rows
#        adapted_columns = columns
#
#    sym_on_right = symmetry_coeff*csr_matrix((WEIGHTS_with_symmetry[adapted_rows],(adapted_rows,adapted_columns)))
#    sym_on_left = symmetry_coeff*csr_matrix((WEIGHTS_with_symmetry[adapted_columns],(adapted_rows,adapted_columns)))
#    sym_on_right_and_left = csr_matrix((WEIGHTS[adapted_rows],(adapted_columns,adapted_columns)))
#    return sym_on_right,sym_on_left,sym_on_right_and_left

#============ CLASS CONTAINING ALL IMPORTANT BLOCKS FOR CORRELATIONS
class blocks_correlations:
    def __init__(self, par, axis, mF, consider_crossed_correlations):
        nb_paths = len(par.paths_to_data)
        factor = 1
        if par.should_we_add_mesh_symmetry:
            factor *= 2
            #weight_sym_on_right,weight_sym_on_left,weight_sym_on_right_and_left = build_symmetrized_weights(par.type_sym,rows,columns,WEIGHTS,WEIGHTS_with_symmetry,
            #D = par.D,axis=axis,mF=mF)

#========== creating necessary blocks
        self.list_blocs = [[None for _ in range(factor*nb_paths)] for _ in range(factor*nb_paths)]

        if consider_crossed_correlations:
            self.list_blocs_crossed = [[None for _ in range(factor*nb_paths)] for _ in range(factor*nb_paths)]

        if par.should_we_penalize_divergence:
            self.list_blocs_dvg = [[None for _ in range(factor*nb_paths)] for _ in range(factor*nb_paths)]
        if consider_crossed_correlations and par.should_we_penalize_divergence:
            self.list_blocs_crossed_dvg = [[None for _ in range(factor*nb_paths)] for _ in range(factor*nb_paths)]

#======================================


    def compute_correlation(self, i, j, par, mF, left_matrix=None, right_matrix=None, weights = None,with_itself = True, crossed_correlations = False):   
        Nx = left_matrix.shape[0]
        type_float = par.type_float
    
        if weights is None:
            weights = csr_matrix(np.ones(Nx),np.arange(Nx),np.arange(Nx))/Nx
        weights = weights.astype(type_float)
    #    output = np.empty((matrix.shape[1], matrix.shape[1]))
        D_eff = par.D + 1*(par.should_we_penalize_divergence) 
        if with_itself:
            rearranged_matrix = rearrange(left_matrix, "(d n) t -> n d t", d = D_eff)[:, :par.D, :]
            matrix_gauss = nodes_to_gauss(rearranged_matrix, par)
            del rearranged_matrix
            matrix_gauss = rearrange(matrix_gauss, "n d t -> (d n) t")
            output = (matrix_gauss.T@weights)@matrix_gauss
    
            if par.should_we_penalize_divergence:
                rearranged_matrix = rearrange(left_matrix, "(d n) t -> n d t", d = D_eff)[:, par.D:par.D+1, :]
    #            matrix_gauss = div(rearranged_matrix, par, par.W, par.R, one_mode = mF)
                matrix_gauss = nodes_to_gauss(rearranged_matrix, par)
                del rearranged_matrix
                matrix_gauss = rearrange(matrix_gauss, "n d t -> (d n) t")
                # print(matrix_gauss.shape, weights.shape, (weights[:weights.shape[0]//par.D]).shape)

                output_dvg = par.penal_div*(matrix_gauss.T@(weights[:weights.shape[0]//par.D, :weights.shape[0]//par.D]))@matrix_gauss
            del matrix_gauss
            gc.collect()
     
        else:
            rearranged_matrix = rearrange(left_matrix, "(d n) t -> n d t", d = D_eff)[:, :par.D, :]
            matrix_gauss = nodes_to_gauss(rearranged_matrix, par)
            rearranged_matrix = rearrange(right_matrix, "(d n) t -> n d t", d = D_eff)[:, :par.D, :]
            second_matrix_gauss = nodes_to_gauss(rearranged_matrix, par)
            del rearranged_matrix
            matrix_gauss = rearrange(matrix_gauss, "n d t -> (d n) t")
            second_matrix_gauss = rearrange(second_matrix_gauss, "n d t -> (d n) t")
            output = (matrix_gauss.T@weights)@second_matrix_gauss
    
            if par.should_we_penalize_divergence:
    
                rearranged_matrix = rearrange(right_matrix, "(d n) t -> n d t", d = D_eff)[:, par.D:par.D+1, :]
    #            matrix_gauss = div(rearranged_matrix, par, par.W, par.R, one_mode = mF)
                matrix_gauss = nodes_to_gauss(rearranged_matrix, par)
                del rearranged_matrix
                matrix_gauss = rearrange(matrix_gauss, "n d t -> (d n) t")
    
                rearranged_matrix = rearrange(left_matrix, "(d n) t-> n d t", d = D_eff)[:, par.D:par.D+1, :]
    #            second_matrix_gauss = div(rearranged_matrix, par, par.W, par.R, one_mode = mF)
                second_matrix_gauss = nodes_to_gauss(rearranged_matrix, par)
                del rearranged_matrix
                second_matrix_gauss = rearrange(second_matrix_gauss, "n d t -> (d n) t")

                output_dvg = par.penal_div*(matrix_gauss.T@(weights[:weights.shape[0]//par.D, :weights.shape[0]//par.D]))@second_matrix_gauss
    
            del matrix_gauss, second_matrix_gauss
            gc.collect()

        if crossed_correlations == False:
            self.list_blocs[i][j] = output

            if par.should_we_penalize_divergence:
                self.list_blocs_dvg[i][j] = output_dvg 

        elif crossed_correlations == True:
            self.list_blocs_crossed[i][j] = output

            if par.should_we_penalize_divergence:
                self.list_blocs_crossed_dvg[i][j] = output_dvg 



#        return output, output_dvg

#def compute_correlation(par, mF, left_matrix=None, right_matrix=None, weights = None,with_itself = True):   
#    Nx = left_matrix.shape[0]
#    type_float = par.type_float
#
#    if weights is None:
#        weights = csr_matrix(np.ones(Nx),np.arange(Nx),np.arange(Nx))/Nx
#    weights = weights.astype(type_float)
##    output = np.empty((matrix.shape[1], matrix.shape[1]))
#
#    if with_itself:
#        rearranged_matrix = rearrange(left_matrix, "(d n) t -> n d t", d = D_eff)[:, :par.D, :]
#        matrix_gauss = nodes_to_gauss(rearranged_matrix, par)
#        del rearranged_matrix
#        matrix_gauss = rearrange(matrix_gauss, "n d t -> (d n) t")
#        output = (matrix_gauss.T@weights)@matrix_gauss
#
#        if par.should_we_penalize_divergence:
#            rearranged_matrix = rearrange(left_matrix, "(d n) t -> n d t", d = D_eff)[:, par.D:par.D+1, :]
##            matrix_gauss = div(rearranged_matrix, par, par.W, par.R, one_mode = mF)
#            matrix_gauss = nodes_to_gauss(rearranged_matrix, par)
#            del rearranged_matrix
#            matrix_gauss = rearrange(matrix_gauss, "n d t -> t (d n)")
#            output_div = par.penal_div(matrix_gauss.T@weights[:weights.shape[0]//3])@matrix_gauss
#        del matrix_gauss
#        gc.collect()
# 
#    else:
#        rearranged_matrix = rearrange(left_matrix, "(d n) t -> n d t", d = D_eff)[:, :par.D, :]
#        matrix_gauss = nodes_to_gauss(rearranged_matrix, par)
#        rearranged_matrix = rearrange(right_matrix, "(d n) t -> n d t", d = D_eff)
#        second_matrix_gauss = nodes_to_gauss(rearranged_matrix, par)
#        del rearranged_matrix
#        matrix_gauss = rearrange(matrix_gauss, "n d t -> (d n) t")
#        second_matrix_gauss = rearrange(second_matrix_gauss, "n d t -> (d n) t")
#        output = (matrix_gauss.T@weights)@second_matrix_gauss
#
#        if par.should_we_penalize_divergence:
#
#            rearranged_matrix = rearrange(right_matrix, "(d n) t -> n d t", d = D_eff)[:, par.D:par.D+1, :]
##            matrix_gauss = div(rearranged_matrix, par, par.W, par.R, one_mode = mF)
#            matrix_gauss = nodes_to_gauss(rearranged_matrix, par)
#            del rearranged_matrix
#            matrix_gauss = rearrange(matrix_gauss, "n d t -> t (d n)")
#
#            rearranged_matrix = rearrange(left_matrix, "t (d n) -> n d t", d = D_eff)
##            second_matrix_gauss = div(rearranged_matrix, par, par.W, par.R, one_mode = mF)
#            second_matrix_gauss = nodes_to_gauss(rearranged_matrix, par)
#            del rearranged_matrix
#
#            output_div = par.penal_div*(matrix_gauss.T@weights[:weights.shape[0]//3])@second_matrix_gauss
#
#        del matrix_gauss, second_matrix_gauss
#        gc.collect()
#    if not par.should_we_penalize_divergence:
#        output_div = None
#
#    return output, output_div


    def symmetrize_blocks(self, par):
        for i in range(len(self.list_blocs)):
            for j in range(len(self.list_blocs)):
                if self.list_blocs[i][j] is None and self.list_blocs[j][i] is None:
                    raise TypeError(f"PB in symmetrize blocks: i={i}, j={j} are lacking")
                if self.list_blocs[i][j] is None:
                    self.list_blocs[i][j] = self.list_blocs[j][i].T
#        for i in range(len(self.list_blocs)):
#            for j in range(len(self.list_blocs)):
#                try:
#                    print(i, j, self.list_blocs[i][j].shape)
#                except AttributeError:
#                    print(self.list_blocs[i][j])#, list_blocs[j][i].shape)
#                    raise AttributeError()
        if par.should_we_penalize_divergence:
            for i in range(len(self.list_blocs_dvg)):
                for j in range(len(self.list_blocs_dvg)):
                    if self.list_blocs_dvg[i][j] is None and self.list_blocs_dvg[j][i] is None:
                        raise TypeError(f"PB in symmetrize blocks: i={i}, j={j} are lacking")
                    if self.list_blocs_dvg[i][j] is None:
                        self.list_blocs_dvg[i][j] = self.list_blocs_dvg[j][i].T

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



def core_correlation_matrix_by_blocks(par,mF,axis,field_name_in_file,for_building_symmetrized_weights=(None,None,None,None),consider_crossed_correlations=False):

    rows,columns,WEIGHTS,WEIGHTS_with_symmetry = for_building_symmetrized_weights

    sparse_WEIGHTS = csr_matrix((WEIGHTS,(np.arange(len(WEIGHTS)),np.arange(len(WEIGHTS))))) # Building a sparse diagonal matrix (required for fct compute_correlation)


    nb_paths = len(par.paths_to_data)
#    factor = 1
#    if par.should_we_add_mesh_symmetry:
#        factor *= 2
#        weight_sym_on_right,weight_sym_on_left,weight_sym_on_right_and_left = build_symmetrized_weights(par.type_sym,rows,columns,WEIGHTS,WEIGHTS_with_symmetry,
#        D = par.D,axis=axis,mF=mF)
#    list_blocs = [[[] for _ in range(factor*nb_paths)] for _ in range(factor*nb_paths)]
#    if consider_crossed_correlations:
#        list_blocs_crossed = [[[] for _ in range(factor*nb_paths)] for _ in range(factor*nb_paths)]
#    else:
#        list_blocs_crossed = None

    all_blocks = blocks_correlations(par, axis, mF, consider_crossed_correlations)
    if par.should_we_add_mesh_symmetry:
        weight_sym_on_right,weight_sym_on_left,weight_sym_on_right_and_left = build_symmetrized_weights(par.type_sym,rows,columns,WEIGHTS,WEIGHTS_with_symmetry,D = par.D,axis=axis,mF=mF) 

    ############### ==============================================================
    ############### importing matrix on left
    ############### ==============================================================

    for i,path_to_data in enumerate(par.paths_to_data):
        first_matrix = import_data(par,mF,axis,path_to_data,field_name_in_file, rm_mean_field=True)
        write_job_output(par,f"      In POD on Fourier => {path_to_data} imported as left matrix")

###################### ============================================================
###################### Computing auto-correlation
###################### ============================================================

#        if par.should_we_add_mesh_symmetry == True:


#            list_blocs[i][i] = compute_correlation(par, mF, left_matrix=first_matrix.T,weights = sparse_WEIGHTS)
#            list_blocs[i][i+factor//2*nb_paths] = compute_correlation(par, mF, left_matrix=first_matrix.T,weights = weight_sym_on_right)
#            list_blocs[i+nb_paths][i+factor//2*nb_paths] = compute_correlation(par, mF, left_matrix=first_matrix.T,weights = weight_sym_on_right_and_left)
            
#            list_blocs[i+nb_paths][i] = list_blocs[i][i+factor//2*nb_paths].T

#        elif par.should_we_add_mesh_symmetry == False:

 #           list_blocs[i][i] = compute_correlation(par, mF, left_matrix=first_matrix.T,weights = sparse_WEIGHTS)

        all_blocks.compute_correlation(i, i, par, mF, left_matrix=first_matrix.T, weights = sparse_WEIGHTS)
        if par.should_we_add_mesh_symmetry:
            all_blocks.compute_correlation(i, i+nb_paths, par, mF, left_matrix=first_matrix.T,weights = weight_sym_on_right)
            all_blocks.compute_correlation(i+nb_paths, i+nb_paths, par, mF, left_matrix=first_matrix.T,weights = weight_sym_on_right_and_left)

    ############### ==============================================================
    ############### importing matrix on right
    ############### ==============================================================

        for j in range(i+1,nb_paths):
            second_matrix = import_data(par,mF,axis,par.paths_to_data[j],field_name_in_file, rm_mean_field=True) #shape t a (d n)
            write_job_output(par,f"          In POD on Fourier => {par.paths_to_data[j]} imported as right matrix")

    ############### ==============================================================
    ############### Computing crossed-correlations (between different paths)
    ############### ==============================================================

            all_blocks.compute_correlation(i, j, par, mF, left_matrix=first_matrix.T,weights = sparse_WEIGHTS,with_itself = False,right_matrix = second_matrix.T)

            if par.should_we_add_mesh_symmetry == True:
                all_blocks.compute_correlation(i, j+nb_paths, par, mF, left_matrix=first_matrix.T,weights = weight_sym_on_right,with_itself = False,right_matrix = second_matrix.T)
                all_blocks.compute_correlation(i+nb_paths, j, par, mF, left_matrix=first_matrix.T,weights = weight_sym_on_left,with_itself = False,right_matrix = second_matrix.T)
                all_blocks.compute_correlation(i+nb_paths, j+nb_paths, par, mF, left_matrix=first_matrix.T,weights = weight_sym_on_right_and_left,with_itself = False,right_matrix = second_matrix.T)
#            if par.should_we_add_mesh_symmetry == True:

#                coeff = -1*(par.type_sym=='Rpi') + 1*(par.type_sym=='centro')

#                list_blocs[i][j] = compute_correlation(par, mF, left_matrix=first_matrix.T,weights = sparse_WEIGHTS,with_itself = False,right_matrix = second_matrix.T)

#                list_blocs[i][j+factor//2*nb_paths] = compute_correlation(par, mF, left_matrix=first_matrix.T,weights = weight_sym_on_right,with_itself = False,right_matrix = second_matrix.T)
#                list_blocs[i+factor//2*nb_paths][j] = compute_correlation(par, mF, left_matrix=first_matrix.T,weights = weight_sym_on_left,with_itself = False,right_matrix = second_matrix.T)
#                list_blocs[i+factor//2*nb_paths][j+factor//2*nb_paths] = compute_correlation(par, mF, left_matrix=first_matrix.T,weights = weight_sym_on_right_and_left,with_itself = False,right_matrix = second_matrix.T)
                
#                list_blocs[j][i] = list_blocs[i][j].T
#                list_blocs[j+factor//2*nb_paths][i] = list_blocs[i][j+factor//2*nb_paths].T
#                list_blocs[j][i+factor//2*nb_paths] = list_blocs[i+factor//2*nb_paths][j].T
#                list_blocs[j+factor//2*nb_paths][i+factor//2*nb_paths] = list_blocs[i+factor//2*nb_paths][j+factor//2*nb_paths].T

#            elif par.should_we_add_mesh_symmetry == False:

#                list_blocs[i][j] = compute_correlation(par, mF, left_matrix=first_matrix.T,weights = sparse_WEIGHTS,with_itself = False,right_matrix = second_matrix.T)

#                list_blocs[j][i] = list_blocs[i][j].T
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
                all_blocks.compute_correlation(i, j, par, mF, left_matrix=first_matrix.T,weights = sparse_WEIGHTS,with_itself = False,right_matrix = second_matrix.T, crossed_correlations = True)
                if par.should_we_add_mesh_symmetry == True:
     #               coeff = -1*(par.type_sym=='Rpi') + 1*(par.type_sym=='centro')
                    coeff = 1
     #============== NO COEFF DUE TO SYMMETRY (cf calculations)

                    all_blocks.compute_correlation(i, j+nb_paths, par, mF, left_matrix=first_matrix.T,weights = coeff*weight_sym_on_right,with_itself = False,right_matrix = second_matrix.T, crossed_correlations = True)
                    all_blocks.compute_correlation(i+nb_paths, j, par, mF, left_matrix=first_matrix.T,weights = weight_sym_on_left,with_itself = False,right_matrix = second_matrix.T, crossed_correlations = True)
                    all_blocks.compute_correlation(i+nb_paths, j+nb_paths, par, mF, left_matrix=first_matrix.T,weights = coeff*weight_sym_on_right_and_left,with_itself = False,right_matrix = second_matrix.T, crossed_correlations = True)
#                if par.should_we_add_mesh_symmetry == True:

#                    coeff = -1*(par.type_sym=='Rpi') + 1*(par.type_sym=='centro')

#                    list_blocs_crossed[i][j] = compute_correlation(par, mF, left_matrix=first_matrix.T,weights = sparse_WEIGHTS,with_itself = False,right_matrix = second_matrix.T)

#                    list_blocs_crossed[i][j+factor//2*nb_paths] = compute_correlation(par, mF, left_matrix=first_matrix.T,weights = coeff*weight_sym_on_right,with_itself = False,right_matrix = second_matrix.T)
#                    list_blocs_crossed[i+factor//2*nb_paths][j] = compute_correlation(par, mF, left_matrix=first_matrix.T,weights = weight_sym_on_left,with_itself = False,right_matrix = second_matrix.T)
#                    list_blocs_crossed[i+factor//2*nb_paths][j+factor//2*nb_paths] = compute_correlation(par, mF, left_matrix=first_matrix.T,weights = coeff*weight_sym_on_right_and_left,with_itself = False,right_matrix = second_matrix.T)

#                elif par.should_we_add_mesh_symmetry == False:

#                    list_blocs_crossed[i][j] = compute_correlation(par, mF, left_matrix=first_matrix.T,weights = sparse_WEIGHTS,with_itself = False,right_matrix = second_matrix.T)
                
            # end for j in [i,nb_paths]
    # end for i,path in paths
#    return list_blocs, list_blocs_crossed

    all_blocks.symmetrize_blocks(par)

    return all_blocks


