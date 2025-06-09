#!/usr/bin/env python3
import os
import gc

# from memory_profiler import profile

#sys.path.append("/ccc/cont003/home/limsi/bousquer/einops")

from einops import rearrange
import numpy as np
from scipy.sparse import csr_matrix
# from basic_functions import write_job_output

class POD:
    def __init__(self,eigvals,proj_coeffs,symmetries):
        self.eigvals = eigvals
        self.proj_coeffs = proj_coeffs
        self.symmetries = symmetries

# @profile
def compute_POD_features(par,correlation,family=None,mF=None,a=None,consider_crossed_correlations=False):
    eigenvalues,eigenvectors = np.linalg.eigh(correlation)
    Nt_float = par.type_float(len(correlation))
    Nt_int = len(correlation)

    del correlation 
    gc.collect()

    if par.number_shifts>1 and not (family is None):
        full_eigenvectors = np.empty((par.number_shifts*eigenvalues.shape[0], eigenvalues.shape[0]), dtype=np.complex128)
        if consider_crossed_correlations:
            if par.should_we_add_mesh_symmetry:
                if par.type_sym == 'Rpi': #these calculations artificially create symmetric/antisymmetric latents (Cf overleaf)
                    sym_vect = (eigenvectors[:Nt_int//2,:] + eigenvectors[Nt_int//2:,:])/2
                    normalization_sym = np.mean(sym_vect.real/sym_vect.imag,axis=0)
                    del sym_vect
                    gc.collect()
                    anti_vect = (eigenvectors[:Nt_int//2,:] - eigenvectors[Nt_int//2:,:])/2
                    normalization_anti = np.mean(anti_vect.real/anti_vect.imag,axis=0)
                    del anti_vect
                    gc.collect()
                    assert np.prod(((normalization_sym+1.j)/(normalization_anti+1.j)).real < 1e-4) == 1
                    eigenvectors /= (normalization_sym+1.j)  #nicely separates sym and antisym
                    eigenvectors /= np.reshape((np.abs(eigenvectors)).sum(0),(1, Nt_int))   #renormalize eigenvectors
    # full_eigenvectors[:Nt, :] = eigenvectors
        for m in range(par.number_shifts):
            full_eigenvectors[m*Nt_int:(m+1)*Nt_int, :] = np.exp(-2*1.j*np.pi*family/par.number_shifts*m)*eigenvectors
        if consider_crossed_correlations:
            real_part = full_eigenvectors.real
            imag_part = full_eigenvectors.imag
            eigenvectors = np.empty((full_eigenvectors.shape[0], 2*full_eigenvectors.shape[1]))
            eigenvectors[:, 0::2] = real_part
            eigenvectors[:, 1::2] = imag_part
            eigenvalues = np.repeat(eigenvalues, 2)
            del real_part, imag_part
            gc.collect()
        else:
            eigenvectors = full_eigenvectors.real
    # else:
    #     eigenvectors = full_eigenvectors.real
        del full_eigenvectors
        gc.collect()
    eigvals = eigenvalues[::-1]
    if eigvals.min() < 0:
        print(f"WARNING: eigenvalues of C_tt have invalid values for {mF} {a}")
    # eigvals = np.abs(eigenvalues[::-1])  # this is good (same signature as code on ruche + the test went well)
    eigenvectors = eigenvectors/(np.sum(eigenvectors**2, axis = 0).reshape(1, eigvals.shape[0]))
    # try:
    proj_coeffs = (np.sqrt(Nt_float*(np.abs(eigvals[:,np.newaxis])).T)*eigenvectors[:,::-1]).T
    # except RuntimeWarning:
    #     proj_coeffs = (np.sqrt(np.abs(Nt_float*(eigvals[:,np.newaxis]).T))*eigenvectors[:,::-1]).T
    #     print(f"WARNING: eigenvalues of C_tt have invalid values for {mF} {a}")
    # if par.rank == 0:
    #     write_job_output(par.path_to_job_output, f'{eigenvectors.shape, proj_coeffs.shape}')
    del eigenvectors
    gc.collect()
    symmetries = np.zeros(eigvals.shape, dtype=np.complex128)
    if par.should_we_add_mesh_symmetry:
        symmetries += np.sign(np.sum(proj_coeffs[:, :Nt_int//2]*proj_coeffs[:, Nt_int//2:Nt_int],axis=1))
    if not (family is None):
        symmetries += 1.j*family

    if not (mF is None):
        if par.type_sym == "Rpi":
            if a == 'c':
                symmetries.real *= 1
            elif a == 's':
                symmetries.real *= -1
        elif par.type_sym == 'centro':
            symmetries.real *= (-1)**mF
        else:
            raise ValueError(f'type_sym must be Rpi or centro, not {par.type_sym}')

    computed_pod = POD(eigvals,proj_coeffs,symmetries)

    return computed_pod 

def save_pod(par,pod_field,is_it_phys_pod=True,family=None,mF=None,fourier_type=None): #matrix is the set of data (necessary when should_we_use_sparse matrices set to True)
        
    Energies=pod_field.eigvals
    latents=pod_field.proj_coeffs
    symmetries = pod_field.symmetries

    if not is_it_phys_pod:
        if fourier_type == "c":
            a = "cos"
        elif fourier_type == "s":
            a = "sin"
            
        if par.should_we_add_mesh_symmetry:
            # latents_not_sym = latents[:len(latents)//2,:len(latents)//2]
            # latents_sym = latents[:len(latents)//2,len(latents)//2:]
            # symmetry_of_latents = np.sign(np.sum(latents_not_sym*latents_sym,axis = 1)) #sum is performed over time at fixed nP
            # if par.type_sym == "Rpi":
            #     if a == 'c':
            #         symmetry_of_latents *= 1
            #     elif a == 's':
            #         symmetry_of_latents *= -1
            # elif par.type_sym == 'centro':
            #     symmetry_of_latents *= (-1)**mF
            # else:
            #     raise ValueError(f'type_sym must be Rpi or centro, not {par.type_sym}')
            if par.type_sym == "Rpi":
                if a == 'c':
                    symmetries.real *= 1
                elif a == 's':
                    symmetries.real *= -1
            elif par.type_sym == 'centro':
                symmetries.real *= (-1)**mF
            else:
                raise ValueError(f'type_sym must be Rpi or centro, not {par.type_sym}')
            os.makedirs(par.complete_output_path+"/"+par.output_file_name+f"/symmetry" ,exist_ok=True)
            np.save(par.complete_output_path+"/"+par.output_file_name+f"/symmetry/{a}_mF{mF:03d}.npy",symmetries)
            # np.save(par.complete_output_path+"/"+par.output_file_name+f"/symmetry/{a}_mF{mF:03d}.npy",symmetry_of_latents)

        os.makedirs(par.complete_output_path+"/"+par.output_file_name+f"/latents" ,exist_ok=True)
        os.makedirs(par.complete_output_path+"/"+par.output_file_name+f"/energies" ,exist_ok=True)
        np.save(par.complete_output_path+"/"+par.output_file_name+f"/latents/{a}_mF{mF:03d}",latents)
        np.save(par.complete_output_path+'/'+par.output_file_name+f'/energies/spectrum_{a}{mF:03d}.npy',Energies)

    else:

        if par.should_we_add_mesh_symmetry:
            # latents_not_sym = latents[:len(latents)//2,:len(latents)//2]
            # latents_sym = latents[:len(latents)//2,len(latents)//2:]
            # symmetry_of_latents = np.sign(np.sum(latents_not_sym*latents_sym,axis = 1)) #sum is performed over time at fixed nP
            if family is None:
                np.save(par.complete_output_path+"/"+par.output_file_name+f"/symmetries_phys.npy",symmetries)
                # np.save(par.complete_output_path+"/"+par.output_file_name+f"/symmetries_phys.npy",symmetry_of_latents)
            else:
                np.save(par.complete_output_path+"/"+par.output_file_name+f"/symmetries_phys_m{family}.npy",symmetries)
                # np.save(par.complete_output_path+"/"+par.output_file_name+f"/symmetries_phys_m{family}.npy",symmetry_of_latents)
        if family is None:
            np.save(par.complete_output_path+"/"+par.output_file_name+f"/a_phys_(mode_time).npy",latents)
            np.save(par.complete_output_path+'/'+par.output_file_name+f'/spectrum_phys.npy',Energies)
        else:
            np.save(par.complete_output_path+"/"+par.output_file_name+f"/a_phys_(mode_time)_m{family}.npy",latents)
            np.save(par.complete_output_path+'/'+par.output_file_name+f'/spectrum_phys_m{family}.npy',Energies)


############ MESH SYMMETRY FUNCTIONS (not working because data too big to be handled by numpy, in practice it is always written explicitly)

def apply_rpi_symmetry(data,D,axis,mF,tab_pairs):

        data = rearrange(data,"t (d n) -> t d n",d=D)
        #real_new_data = np.empty(np.shape(newdata))
        #for d in range(D):
         #   d_coeff = ((d == 0)-1/2)*2 # this coeff has value -1 when d > 1 (so for components theta and z) and 1 when d = 1 (so for component r)
        data = data[:,:,tab_pairs]
        #real_new_data[:,d,:] = d_coeff*newdata[:,d,tab_pairs]
        for d in range(D):
            d_coeff = ((d == 0)-1/2)*2
            data[:,d,:] *= d_coeff
        if axis == 's':  # factor -1 when performing rpi-sym on sine components
            data *= -1
            #real_new_data *= (-1)
        #real_new_data = rearrange(newdata,"t d n  -> t (d n)")
        data = rearrange(data,"t d n  -> t (d n)")
        # return newdata

def apply_centro_symmetry(data,D,axis,mF,tab_pairs):  # THIS IS WRONG BECAUSE IT WILL DEPEND ON THE FOURIER MODE mF

        data = rearrange(data,"t (d n) -> t d n",d=D)
        #real_new_data = np.empty(np.shape(newdata))
        #for d in range(D):
         #   d_coeff = ((d == 0)-1/2)*2 # this coeff has value -1 when d > 1 (so for components theta and z) and 1 when d = 1 (so for component r)
        data = data[:,:,tab_pairs]
        #real_new_data[:,d,:] = d_coeff*newdata[:,d,tab_pairs]
        for d in range(D):
            d_coeff = ((d == 0)+(d == 1)-1/2)*2
            data[:,d,:] *= d_coeff
        if mF%2 == 1:  # factor -1 when performing centro-sym on odd Fourier components
            data *= -1
            #real_new_data *= (-1)
        #real_new_data = rearrange(newdata,"t d n  -> t (d n)")
        data = rearrange(data,"t d n  -> t (d n)")
        # return newdata