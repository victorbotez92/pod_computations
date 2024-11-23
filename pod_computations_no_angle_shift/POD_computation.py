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

class POD:
    def __init__(self,eigvals,proj_coeffs,modes):
        self.eigvals = eigvals
        self.proj_coeffs = proj_coeffs
        self.modes = modes

#@profile

 # 'symmetrized_weights' are necessary when requiring sparse matrices 

def compute_POD_features(correlation):
    eigenvalues,eigenvectors = np.linalg.eigh(correlation)
    Nt = len(correlation)
    del correlation 
    gc.collect()

    eigvals = np.abs(eigenvalues[::-1])  # this is good (same signature as code on ruche + the test went well)
    proj_coeffs = (np.sqrt(Nt*(eigvals[:,np.newaxis]).T)*eigenvectors[:,::-1]).T
    print('eigenvectors',eigenvectors)
    del eigenvectors
    gc.collect()
    computed_pod = POD(eigvals,proj_coeffs,0)

    return computed_pod 


def VB_compute_POD_arrays_snaps_method(matrix,inner_product_weights = [],mF = None,compute_pod_modes = False,
                                       should_we_use_sparse_matrices=False,symmetrized_weights=None,tab_pairs=None):
    
##########################################################
##########################################################
###### Computation of correlation matrix #################
##########################################################
##########################################################
    correlation = compute_correlation_matrix(matrix,inner_product_weights = inner_product_weights,
                                             should_we_use_sparse_matrices=should_we_use_sparse_matrices,
                                             symmetrized_weights=symmetrized_weights)
    # if should_we_use_sparse_matrices == False:
    #     Nt = len(matrix.T)
    #     Nx = len(matrix)
    #     # if len(inner_product_weights) == 0:
    #     #     inner_product_weights = np.ones(len(matrix))*1/Nx
    #     # inner_product_weights = inner_product_weights[:,np.newaxis]
    #     # correlation = (matrix.T*inner_product_weights.T)@matrix*1/Nt
    #     # del inner_product_weights

    # elif should_we_use_sparse_matrices == True:
    #     Nt = 2*len(matrix.T)
    #     Nx = len(matrix)
        # sym_on_right,sym_on_left,sym_on_right_and_left = symmetrized_weights
        # print('inside sparse part of '+"if")
        # bloc_one_one = (matrix.T*inner_product_weights.T)@matrix
        # bloc_one_two = (matrix.T@sym_on_right)@matrix
        # bloc_two_one = (matrix.T@sym_on_left)@matrix
        # bloc_two_two = (matrix.T@sym_on_right_and_left)@matrix

        # correlation = np.block([[bloc_one_one,bloc_one_two],
        #                         [bloc_two_one,bloc_two_two]])*1/Nt

        # del inner_product_weights,bloc_one_one,bloc_one_two,bloc_two_one,bloc_two_two,sym_on_right,sym_on_left,sym_on_right_and_left,symmetrized_weights

##########################################################
##########################################################
##########################################################
##########################################################
##########################################################

    # eigenvalues,eigenvectors = np.linalg.eigh(correlation)
    # del correlation 
    # gc.collect()
    # eigvals = np.abs(eigenvalues[::-1])  # this is good (same signature as code on ruche + the test went well)
    # proj_coeffs = (np.sqrt(Nt*(eigvals[:,np.newaxis]).T)*eigenvectors[:,::-1]).T
    # print('eigenvectors',eigenvectors)
    # del eigenvectors
    # gc.collect()
    # # latents_not_sym = proj_coeffs[:len(proj_coeffs)//2,:]
    # # adapt_signs = np.sum(latents_not_sym,axis=0)/np.abs(1e-13+np.sum(latents_not_sym,axis=0))

    # # proj_coeffs = (proj_coeffs.T*adapt_signs).T



    # if compute_pod_modes and should_we_use_sparse_matrices == False:
    #     modes = (proj_coeffs@matrix.T)/Nt/eigvals[:,np.newaxis]   
    #     del matrix
    #     gc.collect()
    #     computed_pod = POD(eigvals,proj_coeffs,modes.T)

    # else:
    #     computed_pod = POD(eigvals,proj_coeffs,0)

    # return computed_pod
    return compute_POD_features(correlation) 

def save_pod(pod_field,mF,fourier_type,D,output_path,output_file_name,
             is_sym=False,sym_combined=False,is_it_phys_pod=False): #matrix is the set of data (necessary when should_we_use_sparse matrices set to True)
        
    Energies=pod_field.eigvals
    latents=pod_field.proj_coeffs

    if not is_it_phys_pod:

        if fourier_type == "c":
            a = "cos"
        elif fourier_type == "s":
            a = "sin"
        if sym_combined == False:
            if is_sym:
                sym = "_sym"
            else:
                sym = "_not_sym"   
        elif sym_combined == True:
            sym = ""
            
        print("m=",mF,f"{a} eigvals=",Energies)  



        proj_coefficients = pod_field.proj_coeffs
        latents_not_sym = proj_coefficients[:len(proj_coefficients)//2,:len(proj_coefficients)//2]
        latents_sym = proj_coefficients[:len(proj_coefficients)//2,len(proj_coefficients)//2:]

        symmetry_of_latents = np.sign(np.sum(latents_not_sym*latents_sym,axis = 1)) #sum is performed over time at fixed nP
        if a == 'c':
            symmetry_of_latents *= 1
        elif a == 's':
            symmetry_of_latents *= -1

        os.makedirs(output_path+"/"+output_file_name+f"/latents{sym}" ,exist_ok=True)
        os.makedirs(output_path+"/"+output_file_name+f"/energies{sym}" ,exist_ok=True)
        os.makedirs(output_path+"/"+output_file_name+f"/symmetry" ,exist_ok=True)
        np.save(output_path+"/"+output_file_name+f"/latents{sym}/{a}_mF{mF:03d}",latents)
        np.save(output_path+'/'+output_file_name+f'/energies{sym}/spectrum_{a}{mF:03d}.npy',Energies)
        np.save(output_path+"/"+output_file_name+f"/symmetry/{a}_mF{mF:03d}.npy",symmetry_of_latents)

    else:
        print(f"For phys eigvals=",Energies)  
        proj_coefficients = pod_field.proj_coeffs
        latents_not_sym = proj_coefficients[:len(proj_coefficients)//2,:len(proj_coefficients)//2]
        latents_sym = proj_coefficients[:len(proj_coefficients)//2,len(proj_coefficients)//2:]
        symmetry_of_latents = np.sign(np.sum(latents_not_sym*latents_sym,axis = 1)) #sum is performed over time at fixed nP

        np.save(output_path+"/"+output_file_name+f"/a_phys_(mode_time).npy",latents)
        np.save(output_path+'/'+output_file_name+f'/spectrum_phys.npy',Energies)
        np.save(output_path+"/"+output_file_name+f"/symmetries_phys.npy",symmetry_of_latents)

    #save things



def apply_rpi_symmetry(data,D,axis,tab_pairs):

        newdata = rearrange(data,"t (d n) -> t d n",d=D)
        real_new_data = np.empty(np.shape(newdata))
        for d in range(D):
            
            d_coeff = ((d == 0)-1/2)*2 # this coeff has value -1 when d > 1 (so for components theta and z) and 1 when d = 1 (so for component r)

            real_new_data[:,d,:] = d_coeff*newdata[:,d,tab_pairs]

        if axis == 's':  # factor -1 when performing rpi-sym on sine components
            real_new_data *= (-1)

        real_new_data = rearrange(newdata,"t d n  -> t (d n)")
        return newdata