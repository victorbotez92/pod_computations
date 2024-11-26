#!/usr/bin/env python3
import os
import gc

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

def save_pod(par,pod_field,is_it_phys_pod=True,mF=None,fourier_type=None): #matrix is the set of data (necessary when should_we_use_sparse matrices set to True)
        
    Energies=pod_field.eigvals
    latents=pod_field.proj_coeffs

    if not is_it_phys_pod:
        if fourier_type == "c":
            a = "cos"
        elif fourier_type == "s":
            a = "sin"
            
        print("m=",mF,f"{a} eigvals =",Energies)  
        proj_coefficients = pod_field.proj_coeffs

        if par.should_we_add_mesh_symmetry:
            latents_not_sym = proj_coefficients[:len(proj_coefficients)//2,:len(proj_coefficients)//2]
            latents_sym = proj_coefficients[:len(proj_coefficients)//2,len(proj_coefficients)//2:]
            symmetry_of_latents = np.sign(np.sum(latents_not_sym*latents_sym,axis = 1)) #sum is performed over time at fixed nP
            if a == 'c':
                symmetry_of_latents *= 1
            elif a == 's':
                symmetry_of_latents *= -1

            os.makedirs(par.complete_output_path+"/"+par.output_file_name+f"/symmetry" ,exist_ok=True)
            np.save(par.complete_output_path+"/"+par.output_file_name+f"/symmetry/{a}_mF{mF:03d}.npy",symmetry_of_latents)

        os.makedirs(par.complete_output_path+"/"+par.output_file_name+f"/latents" ,exist_ok=True)
        os.makedirs(par.complete_output_path+"/"+par.output_file_name+f"/energies" ,exist_ok=True)
        np.save(par.complete_output_path+"/"+par.output_file_name+f"/latents/{a}_mF{mF:03d}",latents)
        np.save(par.complete_output_path+'/'+par.output_file_name+f'/energies/spectrum_{a}{mF:03d}.npy',Energies)

    else:
        print("Phys eigvals =",Energies)  
        proj_coefficients = pod_field.proj_coeffs

        if par.should_we_add_mesh_symmetry:
            latents_not_sym = proj_coefficients[:len(proj_coefficients)//2,:len(proj_coefficients)//2]
            latents_sym = proj_coefficients[:len(proj_coefficients)//2,len(proj_coefficients)//2:]
            symmetry_of_latents = np.sign(np.sum(latents_not_sym*latents_sym,axis = 1)) #sum is performed over time at fixed nP
            np.save(par.complete_output_path+"/"+par.output_file_name+f"/symmetries_phys.npy",symmetry_of_latents)

        np.save(par.complete_output_path+"/"+par.output_file_name+f"/a_phys_(mode_time).npy",latents)
        np.save(par.complete_output_path+'/'+par.output_file_name+f'/spectrum_phys.npy',Energies)


############ MESH SYMMETRY FUNCTIONS (not working because data too big to be handled by numpy, in practice it is always written explicitly)

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