#!/usr/bin/env python3

#from memory_profiler import profile
import gc

import numpy as np
from scipy.sparse import csr_matrix
from einops import rearrange, einsum

###############################
from functions_to_get_data import import_data
from basic_functions import write_job_output, print_memory_usage
from compute_renormalizations import renormalize_and_rm_mean_field
###############################


###############################
import sys
sys.path.append("/gpfs/users/botezv/.venv")
from SFEMaNS_env.operators import div, nodes_to_gauss
###############################


#============ CLASS CONTAINING ALL IMPORTANT BLOCKS FOR CORRELATIONS
class blocks_correlations:
    def __init__(self, inputs):#, axis, mF):
        nb_paths = len(inputs.paths_to_data)
        factor = 1
        if inputs.should_we_add_mesh_symmetry:
            factor *= 2

#========== creating necessary blocks
        self.list_blocs = [[None for _ in range(factor*nb_paths)] for _ in range(factor*nb_paths)]

        # if consider_crossed_correlations:
        #     self.list_blocs_crossed = [[None for _ in range(factor*nb_paths)] for _ in range(factor*nb_paths)]

        # if inputs.should_we_penalize_divergence:
        #     self.list_blocs_dvg = [[None for _ in range(factor*nb_paths)] for _ in range(factor*nb_paths)]
        # if consider_crossed_correlations and inputs.should_we_penalize_divergence:
        #     self.list_blocs_crossed_dvg = [[None for _ in range(factor*nb_paths)] for _ in range(factor*nb_paths)]

#======================================


    def compute_correlation(self, i, j, inputs, global_coeff=1, left_matrix=None, right_matrix=None, symmetrize = False):   
     
        left_matrix_gauss = nodes_to_gauss(left_matrix, inputs)
        if right_matrix is None:
            right_matrix_gauss = np.copy(left_matrix_gauss)
        else:
            right_matrix_gauss = nodes_to_gauss(right_matrix, inputs)
        if symmetrize:
            # global_coeff = 1
            # if inputs.type_sym == 'Rpi' and a == 's':
            #     global_coeff = -1
            # if inputs.type_sym == 'centro' and mF%2==1:
            #     global_coeff = -1
            right_matrix_gauss = (inputs.sym_signs.reshape(1, inputs.sym_signs.shape[0], 1))*right_matrix_gauss[inputs.tab_pairs, :, :]
        output = einsum(left_matrix_gauss, global_coeff*right_matrix_gauss, inputs.W, "n D T, n D T_prime, n -> T T_prime")

        del left_matrix_gauss, right_matrix_gauss
        gc.collect()

        self.list_blocs[i][j] = output

        # if crossed_correlations == False:
        #     self.list_blocs[i][j] = output

        #     if inputs.should_we_penalize_divergence:
        #         self.list_blocs_dvg[i][j] = output_dvg 

        # elif crossed_correlations == True:
        #     self.list_blocs_crossed[i][j] = output

        #     if inputs.should_we_penalize_divergence:
        #         self.list_blocs_crossed_dvg[i][j] = output_dvg 



    def symmetrize_blocks(self, inputs):
        for i in range(len(self.list_blocs)):
            for j in range(len(self.list_blocs)):
                if self.list_blocs[i][j] is None and self.list_blocs[j][i] is None:
                    raise TypeError(f"PB in symmetrize blocks: i={i}, j={j} are lacking")
                if self.list_blocs[i][j] is None:
                    self.list_blocs[i][j] = self.list_blocs[j][i].T

        if inputs.should_we_penalize_divergence:
            for i in range(len(self.list_blocs_dvg)):
                for j in range(len(self.list_blocs_dvg)):
                    if self.list_blocs_dvg[i][j] is None and self.list_blocs_dvg[j][i] is None:
                        raise TypeError(f"PB in symmetrize blocks: i={i}, j={j} are lacking")
                    if self.list_blocs_dvg[i][j] is None:
                        self.list_blocs_dvg[i][j] = self.list_blocs_dvg[j][i].T




def core_correlation_matrix_by_blocks(inputs,mF,axis,consider_crossed_correlations=False):


    a = 0*(axis=='c') + 1*(axis=='s')
    counter_axis = ["c", "s"][1-a]
    nb_paths = len(inputs.paths_to_data)

    all_blocks = blocks_correlations(inputs)
    if consider_crossed_correlations:
        all_blocks_crossed = blocks_correlations(inputs)
        eps_coeff = 1*(axis=='c') + (-1)*(axis=='s')

    else:
        all_blocks_crossed = None

    if inputs.should_we_add_mesh_symmetry:
        sym_coeff = 1
        if inputs.type_sym == 'Rpi' and axis == 's':
            sym_coeff = -1
        if inputs.type_sym == 'centro' and mF%2==1:
            sym_coeff = -1
        
        if consider_crossed_correlations:
            sym_coeff_cor = 1
            if inputs.type_sym == 'Rpi' and axis == 'c': #opposite sign when Rpi symmetry
                sym_coeff_cor = -1
            if inputs.type_sym == 'centro' and mF%2==1: #same sign when centro-symmetry
                sym_coeff_cor = -1
            

    ############### ==============================================================
    ############### importing matrix on left
    ############### ==============================================================
    for i in range(nb_paths):
        print_memory_usage(inputs, tag="Before importing left matrix")

        first_matrix = import_data(inputs,mF,inputs.paths_to_data[i])[:, a::2, :] # shape n d t
        renormalize_and_rm_mean_field(inputs, first_matrix, mF, axis)

        write_job_output(inputs,f"      In POD on Fourier => {inputs.paths_to_data[i]} imported as left matrix")

        print_memory_usage(inputs, tag="After importing left matrix")

###################### ============================================================
###################### Computing auto-correlation
###################### ============================================================

        all_blocks.compute_correlation(i, i, inputs, left_matrix=first_matrix)#, weights = sparse_WEIGHTS)

        if inputs.should_we_add_mesh_symmetry:
            all_blocks.compute_correlation(i, i+nb_paths, inputs, global_coeff=sym_coeff, left_matrix=first_matrix, symmetrize=True)#,weights = weight_sym_on_right)
            all_blocks.compute_correlation(i+nb_paths, i+nb_paths, inputs, left_matrix=first_matrix)#, symmetrize=True)#,weights = weight_sym_on_right_and_left)

    ############### ==============================================================
    ############### importing matrix on right
    ############### ==============================================================

        for j in range(i+1,nb_paths):
            print_memory_usage(inputs, tag="Before importing right matrix:")

            second_matrix = import_data(inputs,mF,inputs.paths_to_data[j])[:, a::2, :] #shape n d t
            renormalize_and_rm_mean_field(inputs, second_matrix, mF, axis)
            write_job_output(inputs,f"          In POD on Fourier => {inputs.paths_to_data[j]} imported as right matrix")
            print_memory_usage(inputs, tag="After importing right matrix:")

    ############### ==============================================================
    ############### Computing crossed-correlations (between different paths)
    ############### ==============================================================

            all_blocks.compute_correlation(i, j, inputs, left_matrix=first_matrix, right_matrix=second_matrix)#,weights = sparse_WEIGHTS,with_itself = False,right_matrix = second_matrix.T)

            if inputs.should_we_add_mesh_symmetry == True:
                all_blocks.compute_correlation(i, j+nb_paths, inputs, global_coeff=sym_coeff, left_matrix=first_matrix,right_matrix = second_matrix, symmetrize=True)
                all_blocks.compute_correlation(i+nb_paths, j, inputs, global_coeff=sym_coeff, left_matrix=first_matrix,right_matrix = second_matrix, symmetrize=True)
                all_blocks.compute_correlation(i+nb_paths, j+nb_paths, inputs, left_matrix=first_matrix,right_matrix = second_matrix)

            print_memory_usage(inputs, tag="After computing correlations:")

        # end for j in [i+1,nb_paths]
    ############### ==============================================================
    ############### importing matrix on right
    ############### ==============================================================
        if consider_crossed_correlations:
            for j in range(nb_paths):
                print_memory_usage(inputs, tag="Before importing right crossed matrix:")
                second_matrix = import_data(inputs,mF,inputs.paths_to_data[j])[:, (1-a)::2, :] #shape t a (d n)
                renormalize_and_rm_mean_field(inputs, second_matrix, mF, counter_axis)
                write_job_output(inputs,f"          In crossed POD on Fourier => {inputs.paths_to_data[j]} imported as right matrix")
                print_memory_usage(inputs, tag="After importing right crossed matrix:")

        ############### ==============================================================
        ############### Computing crossed-correlations (between cos and sine)
        ############### ==============================================================
                all_blocks_crossed.compute_correlation(i, j, inputs, global_coeff=eps_coeff, left_matrix=first_matrix, right_matrix = second_matrix)
                if inputs.should_we_add_mesh_symmetry == True:
# TEST 10/10/2025
                    all_blocks_crossed.compute_correlation(i, j+nb_paths, inputs, global_coeff=eps_coeff*sym_coeff_cor, left_matrix=first_matrix, right_matrix = second_matrix, symmetrize = True)
                    all_blocks_crossed.compute_correlation(i+nb_paths, j, inputs, global_coeff=eps_coeff*sym_coeff, left_matrix=first_matrix,right_matrix = second_matrix, symmetrize = True)
                    all_blocks_crossed.compute_correlation(i+nb_paths, j+nb_paths, inputs, global_coeff=eps_coeff*sym_coeff*sym_coeff_cor, left_matrix=first_matrix, right_matrix = second_matrix)
                
                print_memory_usage(inputs, tag="After computing crossed correlations:")

            # end for j in [i,nb_paths]
    # end for i,path in paths

    all_blocks.symmetrize_blocks(inputs)
    
    return all_blocks, all_blocks_crossed


