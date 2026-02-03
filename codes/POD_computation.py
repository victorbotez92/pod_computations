#!/usr/bin/env python3
import os,sys
import gc

# from memory_profiler import profile

#sys.path.append("/ccc/cont003/home/limsi/bousquer/einops")

from einops import rearrange
import numpy as np
from scipy.sparse import csr_matrix

from functions_to_get_data import import_mean_field
from compute_renormalizations import renormalize_and_rm_mean_field

sys.path.append("/gpfs/users/botezv/.venv")
from SFEMaNS_env.operators import nodes_to_gauss, gauss_to_nodes
class POD:
    def __init__(self,eigvals,proj_coeffs,symmetries):
        self.eigvals = eigvals
        self.proj_coeffs = proj_coeffs
        self.symmetries = symmetries

def inv_vec(vec):
    n = len(vec)
    return np.concatenate((vec[n//2:, :], vec[:n//2, :]))

# @profile
def compute_POD_features(inputs,correlation,family=None,mF=None,axis=None,consider_crossed_correlations=False):
    # artificially choose to add factor 1/sqrt(2) to deal with ||u + i tilde(u)||^2 = 2||u||^2
    # if consider_crossed_correlations:
    #     eigenvalues,eigenvectors = np.linalg.eigh(2*correlation)
    # else:
    #     eigenvalues,eigenvectors = np.linalg.eigh(correlation)
    if consider_crossed_correlations:
        eigenvalues,eigenvectors = np.linalg.eigh(1/2*correlation)
    else:
        eigenvalues,eigenvectors = np.linalg.eigh(correlation)


    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]

    Nt_float = inputs.type_float(len(correlation))
    Nt_int = len(correlation)

    del correlation 
    gc.collect()

    if inputs.number_shifts>1 and not (family is None):
        full_eigenvectors = np.empty((inputs.number_shifts*eigenvalues.shape[0], eigenvalues.shape[0]), dtype=np.complex128)
        if consider_crossed_correlations:
            if inputs.should_we_add_mesh_symmetry and inputs.type_sym == "Rpi":
                # if inputs.type_sym == 'Rpi': #these calculations artificially create symmetric/antisymmetric latents (Cf overleaf)
                # sym_vect = (eigenvectors[:Nt_int//2,:] + eigenvectors[Nt_int//2:,:])/2
                # normalization_sym = np.mean(sym_vect.real/sym_vect.imag,axis=0)
                # del sym_vect
                # gc.collect()
                # anti_vect = (eigenvectors[:Nt_int//2,:] - eigenvectors[Nt_int//2:,:])/2
                # normalization_anti = np.mean(anti_vect.real/anti_vect.imag,axis=0)
                # del anti_vect
                # gc.collect()

                # if np.prod(((normalization_sym+1.j)/(normalization_anti+1.j)).real/((normalization_sym+1.j)/(normalization_anti+1.j)).imag < 1e-4) != 1:
                #     print(f'WARNING IN POD_computation.py {family}-family: not satisfactory separation between antisymmetric and symmetric')

                angle_S = np.angle((eigenvectors+inv_vec(eigenvectors))/2)
                angle_A = np.angle((eigenvectors-inv_vec(eigenvectors))/2/1.j)


                if np.prod(np.mean((angle_S-angle_A)%(np.pi)/np.pi, axis=0)<1e-4) != 1:
                    print(f'WARNING IN POD_computation.py {family}-family: not satisfactory separation between antisymmetric and symmetric')

                eigenvectors *= np.exp(-1.j*angle_S).mean(axis=0).reshape(1, eigenvectors.shape[1])
                del angle_S
                gc.collect()

                #assert np.prod(((normalization_sym+1.j)/(normalization_anti+1.j)).real/((normalization_sym+1.j)/(normalization_anti+1.j)).imag < 1e-4) == 1
                # eigenvectors /= (normalization_sym+1.j)  #nicely separates sym and antisym
                eigenvectors /= np.reshape(np.sqrt((np.abs(eigenvectors**2)).sum(0)),(1, Nt_int))   #renormalize eigenvectors

        for m in range(inputs.number_shifts):
            full_eigenvectors[m*Nt_int:(m+1)*Nt_int, :] = np.exp(-2*1.j*np.pi*family/inputs.number_shifts*m)*eigenvectors
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
        del full_eigenvectors
        gc.collect()
    else:
        eigenvectors = eigenvectors.real
        #the following two lines make sure a mean field component will have positive time-evolution
        new_sign = np.sign(eigenvectors[:eigenvalues.shape[0]//inputs.number_shifts//4,:].sum(axis=0))
        eigenvectors *= new_sign.reshape(1, new_sign.shape[0])
    if eigenvalues.min() < 0:
        print(f"WARNING: eigenvalues of C_tt have invalid values for {mF} {axis}")
    
    eigenvectors = eigenvectors/np.sqrt(np.sum(np.abs(eigenvectors)**2, axis = 0)).reshape(1, eigenvectors.shape[1])
    proj_coeffs = (np.sqrt(Nt_float*inputs.number_shifts*(np.abs(eigenvalues[:,np.newaxis])).T)*eigenvectors).T    #factor number_shifts required because Nt_float is just the raw size of snapshot matrix
    del eigenvectors
    gc.collect()
    symmetries = np.zeros(eigenvalues.shape, dtype=np.complex128)
    if inputs.should_we_add_mesh_symmetry:
        symmetries += np.sign(np.sum(proj_coeffs[:, :Nt_int//2]*proj_coeffs[:, Nt_int//2:Nt_int],axis=1))
    if not (family is None):
        symmetries += 1.j*family

    if not (mF is None):
        if inputs.type_sym == "Rpi":
            if axis == 'c':
                symmetries.real *= 1
            elif axis == 's':
                symmetries.real *= -1
        elif inputs.type_sym == 'centro':
            symmetries.real *= (-1)**mF
        else:
            raise ValueError(f'type_sym must be Rpi or centro, not {inputs.type_sym}')

    computed_pod = POD(eigenvalues,proj_coeffs,symmetries)

    return computed_pod 

        
def update_pod_with_mean_field(inputs,pod_field,is_it_phys_pod=True,family=None,mF=None,axis=None): #matrix is the set of data (necessary when should_we_use_sparse matrices set to True)
        
    Energies=pod_field.eigvals
    latents=pod_field.proj_coeffs
    symmetries = pod_field.symmetries

    if is_it_phys_pod == False:
#        if fourier_type == "c":
#            a = "cos"
#        elif fourier_type == "s":
#            a = "sin"
            
#============= INCORPORATING MEAN FIELD TO DATA
        if inputs.should_we_remove_mean_field:
            bool_import_mean_field = (mF == 0 and axis=='c') or ((mF != 0) and (mF%inputs.number_shifts==0) and (inputs.should_mean_field_be_axisymmetric == False))
            if bool_import_mean_field:
                #mean_field = np.zeros((1,1,1))
                mean_field = np.load(f"{inputs.MF_output}/mF{mF}_{axis}.npy")
                #renormalize_and_rm_mean_field(inputs, mean_field, mF, axis)
                #mean_field *= -1
         #       _, _, WEIGHTS, _ = inputs.for_building_symmetrized_weights
                mean_field = mean_field.reshape(mean_field.shape[0], mean_field.shape[1], 1) 
                mean_energy = np.sum(nodes_to_gauss(mean_field**2, inputs)*inputs.W.reshape(inputs.W.shape[0], 1, 1))
                Energies = np.concatenate((np.asarray([mean_energy]), Energies))          
    
                cst_latent = np.sqrt(mean_energy)*np.ones(latents.shape[1])
                cst_latent = cst_latent.reshape(1, cst_latent.shape[0])
                latents = np.vstack((cst_latent, latents))
    
                if inputs.should_we_add_mesh_symmetry:
                    if inputs.type_sym == "Rpi":
                        if axis == 'c':
                            sym_mean_field = 1+mF*1.j
                        elif axis == 's':
                            sym_mean_field = -1-mF*1.j
                    elif inputs.type_sym == 'centro':
                        sym_mean_field = (-1)**mF*(1+mF*1.j)
                    else:
                        raise ValueError(f'type_sym must be Rpi or centro, not {inputs.type_sym}')
                    symmetries = np.concatenate((np.asarray([sym_mean_field]), symmetries)) 
#============= INCORPORATING MEAN FIELD TO DATA

    else:

#============= INCORPORATING MEAN FIELD TO DATA
        if inputs.should_we_remove_mean_field:
            bool_import_mean_field = (not family is None) and (family == 0)
            if bool_import_mean_field:
                mean_energy = 0
                for mF in (inputs.list_m_families[0]):
                    for axis_import in ['c','s']:
                        if (mF==0 and axis_import=='s') or (mF!=0 and inputs.should_mean_field_be_axisymmetric==False):
                            continue
                        mean_field = np.load(f"{inputs.MF_output}/mF{mF}_{axis_import}.npy")
                        mean_field = mean_field.reshape(mean_field.shape[0], mean_field.shape[1], 1) 
                        #mean_field = np.zeros((1, 1, 1))
                        #renormalize_and_rm_mean_field(par, mean_field, mF, axis)
                        #mean_field *= -1
                #        mean_field = import_mean_field(inputs, mF, axis_import)
              #          _, _, WEIGHTS, _ = inputs.for_building_symmetrized_weights
                        fourier_factor = 1/2*(mF > 0) + 1*(mF == 0)

                        mean_energy += fourier_factor*np.sum(nodes_to_gauss(mean_field**2, inputs)*inputs.W.reshape(inputs.W.shape[0], 1, 1))

                #Energies = np.concatenate((np.array([mean_energy]), Energies))          

                #cst_latent = np.sqrt(mean_energy)/(inputs.number_shifts/latents.shape[1])*np.ones(latents.shape[1])
                #cst_latent = cst_latent.reshape(1, cst_latent.shape[0])

                cst_latent = np.sqrt(mean_energy)*np.ones(latents.shape[1])
                cst_latent = cst_latent.reshape(1, cst_latent.shape[0])
                latents = np.vstack((cst_latent, latents))
                Energies = np.concatenate((np.asarray([mean_energy]), Energies))          
              
                if inputs.should_we_add_mesh_symmetry:
                    symmetries = np.concatenate((np.asarray([1+0.j]), symmetries))
#                print("cst latent is = ", cst_latent.mean(), cst_latent.std())
#                print("mean_energy is =", mean_energy)
#                print(1/0) 
#============= INCORPORATING MEAN FIELD TO DATA
    pod_field.eigvals=Energies
    pod_field.proj_coeffs=latents
    pod_field.symmetries=symmetries
    
    if Energies.min() == 0:
        raise ValueError(f"Energy Error in update_pod_with_mean_field: {pod_field},from_phys={is_it_phys_pod},family={family},mF={mF},fourier_type={a})")
    if np.abs(latents[0]).min() == 0:
        raise ValueError(f"new latent Error in update_pod_with_mean_field: {pod_field},from_phys={is_it_phys_pod},family={family},mF={mF},fourier_type={a})")
    #if np.abs(latents).min() == 0:
     #   raise ValueError(f"latent Error in update_pod_with_mean_field: {pod_field},from_phys={is_it_phys_pod},family={family},mF={mF},fourier_type={fourier_type})")
    return pod_field

def save_pod(inputs,pod_field,is_it_phys_pod=True,family=None,mF=None,axis=None): #matrix is the set of data (necessary when should_we_use_sparse matrices set to True)
        
    Energies=pod_field.eigvals
    latents=pod_field.proj_coeffs
    symmetries = pod_field.symmetries

    if is_it_phys_pod == False:
#        if fourier_type == "c":
#            a = "cos"
#        elif fourier_type == "s":
#            a = "sin"
            
        if inputs.should_we_add_mesh_symmetry:
            if inputs.type_sym == "Rpi":
                if axis == 'c':
                    symmetries.real *= 1
                elif axis == 's':
                    symmetries.real *= -1
            elif inputs.type_sym == 'centro':
                symmetries.real *= (-1)**mF
            else:
                raise ValueError(f'type_sym must be Rpi or centro, not {inputs.type_sym}')
            os.makedirs(inputs.complete_output_path+"/"+inputs.output_file_name+f"/symmetry" ,exist_ok=True)
            np.save(inputs.complete_output_path+"/"+inputs.output_file_name+f"/symmetry/{axis}_mF{mF:03d}.npy",symmetries)

        os.makedirs(inputs.complete_output_path+"/"+inputs.output_file_name+f"/latents" ,exist_ok=True)
        os.makedirs(inputs.complete_output_path+"/"+inputs.output_file_name+f"/energies" ,exist_ok=True)
        np.save(inputs.complete_output_path+"/"+inputs.output_file_name+f"/latents/{axis}_mF{mF:03d}",latents)
        np.save(inputs.complete_output_path+'/'+inputs.output_file_name+f'/energies/spectrum_{axis}{mF:03d}.npy',Energies)

    else:

        if inputs.should_we_add_mesh_symmetry:
            if family is None:
                np.save(inputs.complete_output_path+"/"+inputs.output_file_name+f"/symmetries_phys.npy",symmetries)
            else:
                np.save(inputs.complete_output_path+"/"+inputs.output_file_name+f"/symmetries_phys_m{family}.npy",symmetries)
        if family is None:
            np.save(inputs.complete_output_path+"/"+inputs.output_file_name+f"/a_phys.npy",latents)
            np.save(inputs.complete_output_path+'/'+inputs.output_file_name+f'/spectrum_phys.npy',Energies)
        else:
            np.save(inputs.complete_output_path+"/"+inputs.output_file_name+f"/a_phys_m{family}.npy",latents)
            np.save(inputs.complete_output_path+'/'+inputs.output_file_name+f'/spectrum_phys_m{family}.npy',Energies)


############ MESH SYMMETRY FUNCTIONS (not working because data too big to be handled by numpy, in practice it is always written explicitly)

def apply_3D_mesh_sym(inputs, data):
    data[:] = data[inputs.tab_pairs, :, :]
    if inputs.type_sym == 'Rpi': #-1 for sine types
        data[:, 0::2, :] *= inputs.sym_signs.reshape(1, inputs.D, 1)
        data[:, 1::2, :] *= -inputs.sym_signs.reshape(1, inputs.D, 1)
    elif inputs.type_sym == 'centro': #-1 for odd mF
        data[:, 0::2, :] *= inputs.sym_signs.reshape(1, inputs.D, 1)
        data[:, 1::2, :] *= inputs.sym_signs.reshape(1, inputs.D, 1)
        data[:, :, 1::2] *= -1

def apply_mesh_sym_at_mF(inputs, data, mF):
    data[:] = data[inputs.tab_pairs, :, :]
    if inputs.type_sym == 'Rpi': #-1 for sine types
        data[:, 0::2, :] *= inputs.sym_signs.reshape(1, inputs.D, 1)
        data[:, 1::2, :] *= -inputs.sym_signs.reshape(1, inputs.D, 1)
    elif inputs.type_sym == 'centro': #-1 for odd mF
        data[:, 0::2, :] *= inputs.sym_signs.reshape(1, inputs.D, 1)
        data[:, 1::2, :] *= inputs.sym_signs.reshape(1, inputs.D, 1)
        data[:, :, :] *= -1

def apply_mesh_sym(inputs, data, axis, mF):
    if inputs.type_sym == 'Rpi':
        apply_rpi_symmetry(inputs, data, axis)
    elif inputs.type_sym == 'centro':
        apply_centro_symmetry(inputs, data, mF)


def apply_rpi_symmetry(inputs, data, axis):
    data[:] = data[inputs.tab_pairs,:,:]
    data *= inputs.sym_signs.reshape(1, inputs.D, 1)
    if axis == 's':
        data *= -1

def apply_centro_symmetry(inputs, data, mF):  # THIS IS WRONG BECAUSE IT WILL DEPEND ON THE FOURIER MODE mF
    data[:] = data[inputs.tab_pairs,:,:]
    data *= inputs.sym_signs.reshape(1, inputs.D, 1)
    if mF%2 == 1:
        data *= -1

def coeff_sym_axis(inputs, mF, axis):
    coeff = 1
    if inputs.type_sym == 'Rpi' and axis=='s':
        coeff = -1
    if inputs.type_sym == 'centro' and mF%2 == 1:
        coeff = -1
    return coeff
