import sys
import numpy as np
from einops import rearrange, einsum

from basic_functions import write_job_output
from functions_to_get_data import import_data
from compute_renormalizations import apply_3D_mesh_sym
#from POD_computation import apply_3D_mesh_sym



sys.path.append("/gpfs/users/botezv/.venv/")
from SFEMaNS_env.read_stb import get_fourier
from SFEMaNS_env.FFT_operations import FFT_EUCLIDIAN_PROD
from SFEMaNS_env.operators import nodes_to_gauss, div



def main_test(inputs, sfem_par):
    write_job_output(inputs, "ORTHONORMALIZATION TESTS")
    test_normalization(inputs, sfem_par)
    if inputs.size != 1:
        inputs.comm.Barrier()
    
    write_job_output(inputs, "DIVERGENCE TESTS")
    test_divergence(inputs, sfem_par)
    if inputs.size != 1:
        inputs.comm.Barrier()
    
    write_job_output(inputs, "POD CUMSUM TESTS")
    test_cumulated_sum(inputs, sfem_par)
    if inputs.size != 1:
        inputs.comm.Barrier()

    write_job_output(inputs, "LATENTS CUMSUM TESTS")
    test_cumulated_energy(inputs, sfem_par)
    if inputs.size != 1:
        inputs.comm.Barrier()

    write_job_output(inputs, "SYMMETRY TESTS")
    test_cumulated_energy(inputs, sfem_par)
    if inputs.size != 1:
        inputs.comm.Barrier()


def test_normalization(inputs, sfem_par):
    path_bins = inputs.complete_output_path+inputs.output_file_name+f"/phys_pod_modes/"
    path_out = inputs.complete_output_path+inputs.output_file_name+f"/post_tests/"

    for m_family in inputs.list_m_families:
        m = m_family.min()%inputs.number_shifts
        write_job_output(inputs,f"      Starting orthonormality test for {m}-family")

        sfem_par.add_bins(path_bins+f"m{m:03d}/", field=f'POD_{inputs.field}', D=inputs.D, replace=True)

        correlation_matrix = np.zeros((inputs.phys_pod_modes_to_save.shape[0], inputs.phys_pod_modes_to_save.shape[0]))
        # list_pair_modes = [None for _ in range(inputs.phys_pod_modes_to_save.shape[0]*(inputs.phys_pod_modes_to_save.shape[0]+1)/2)]
        list_pair_modes = []
        for n1 in inputs.phys_pod_modes_to_save:
            for n2 in inputs.phys_pod_modes_to_save:
                list_pair_modes.append((n1, n2))

        # list_pair_modes = [(n1, n2) for n1 in inputs.phys_pod_modes_to_save for n2 in inputs.phys_pod_modes_to_save]
        old_n1 = -1

        computations_per_proc = max(len(list_pair_modes)//inputs.size,1)
        departure_at_rank_k = computations_per_proc*inputs.rank

        for i in range(departure_at_rank_k, min(departure_at_rank_k+computations_per_proc, len(list_pair_modes))):
        # for i in range(inputs.rank, len(list_pair_modes), inputs.size):
            i1, i2 = list_pair_modes[i]
            n1, n2 = inputs.phys_pod_modes_to_save[i1], inputs.phys_pod_modes_to_save[i2]
            print(n1, n2)
            if n1 != old_n1:
                phi_1 = get_fourier(sfem_par, I=n1+1)
            if n1 != n2:
                phi_2 = get_fourier(sfem_par, I=n2+1)
                result = FFT_EUCLIDIAN_PROD(nodes_to_gauss(phi_1, inputs), nodes_to_gauss(phi_2, inputs), inputs.W)
            elif n1 == n2:
                result = FFT_EUCLIDIAN_PROD(nodes_to_gauss(phi_1, inputs), nodes_to_gauss(phi_1, inputs), inputs.W)
            correlation_matrix[i1, i2] = result
            correlation_matrix[i2, i1] = result

            old_n1 = n1

        correlation_matrix = inputs.comm.reduce(correlation_matrix, root=0)
        if inputs.rank == 0:
            print(correlation_matrix)
            correlation_matrix /= 2*np.pi
            bool_normal = True
            bool_orthogonal = True
            for i in range(correlation_matrix.shape[0]):
                for j in range(correlation_matrix.shape[1]):
                    if np.abs(correlation_matrix[i,j]) > 1e-7 and i!=j:
                        write_job_output(inputs, str(([i,j], correlation_matrix[i,j])))
                        bool_orthogonal = False
                    elif (np.abs(correlation_matrix[i,j]-1)>1e-7) and i==j:
                        write_job_output(inputs, str(([i,j], correlation_matrix[i,j])))
                        bool_normal = False
            if bool_normal:
                write_job_output(inputs, f"            Normality Test successful for {m}-family")
            else:
                write_job_output(inputs, f"            Normality Test failed for {m}-family")
            if bool_orthogonal:
                write_job_output(inputs, f"            Orthogonality Test successful for {m}-family")
            else:
                write_job_output(inputs, f"            Orthogonality Test failed for {m}-family")
            np.save(path_out+f"orthonormality_{m}.npy", correlation_matrix)




def test_divergence(inputs, sfem_par, eps_dvg=1e-2):
    path_bins = inputs.complete_output_path+inputs.output_file_name+f"/phys_pod_modes/"
    path_out = inputs.complete_output_path+inputs.output_file_name+f"/post_tests/"

    for m_family in inputs.list_m_families:
        m = m_family.min()%inputs.number_shifts
        write_job_output(inputs,f"      Starting divergence test for {m}-family")

        sfem_par.add_bins(path_bins+f"m{m:03d}/", field=f'POD_{inputs.field}', D=inputs.D, replace=True)

        divergence_matrix = np.zeros(inputs.phys_pod_modes_to_save.shape[0])
        bool_dvg = True
        for i in range(inputs.rank, inputs.phys_pod_modes_to_save.shape[0], inputs.size):
            nP = inputs.phys_pod_modes_to_save[i]

            phi = get_fourier(sfem_par, I=nP+1)
            divergence = div(phi, inputs)
            result = FFT_EUCLIDIAN_PROD(divergence, divergence, inputs.W)
            divergence_matrix[i] = result
        divergence_matrix = inputs.comm.reduce(divergence_matrix, root=0)
        if inputs.rank == 0:
            for i in range(divergence_matrix.shape[0]):
                res = divergence_matrix[i]
                if res/2/np.pi > eps_dvg:
                    bool_dvg = False
                    write_job_output(inputs, str((i, res/2/np.pi)))
            if bool_dvg:
                write_job_output(inputs, f"            Relative Divergence Test successful for {m}-family")
            else:
                write_job_output(inputs, f"            Relative Divergence Test failed for {m}-family")
            
        np.save(path_out+f"divergence_{m}.npy", divergence_matrix)


def test_cumulated_sum(inputs, sfem_par):
    path_bins = inputs.complete_output_path+inputs.output_file_name+f"/phys_pod_modes/"
    path_latents = inputs.complete_output_path+inputs.output_file_name
    path_out = inputs.complete_output_path+inputs.output_file_name+f"/post_tests/"
    nb_paths = len(inputs.paths_to_data)

    for m_family in inputs.list_m_families:
        m = m_family.min()%inputs.number_shifts
        write_job_output(inputs,f"      Starting cumulated sum test for {m}-family")

        sfem_par.add_bins(path_bins+f"m{m:03d}/", field=f'POD_{inputs.field}', D=inputs.D, replace=True)
        latents = np.load(path_latents+f"a_phys_m{m}.npy")

        energy_difference_matrix = np.zeros((inputs.phys_pod_modes_to_save.shape[0]+1, np.sum(inputs.list_T)))

        for i,mF in enumerate(inputs.list_modes[inputs.rank_fourier::inputs.nb_proc_in_fourier]):
            for a in range(inputs.rank_axis,2,inputs.nb_proc_in_axis):
                if (mF == 0 and a == 1) or (not mF in m_family):
                    continue

                #energy_difference_matrix = np.zeros((inputs.phys_pod_modes_to_save.shape[0]+1, inputs.list_T.sum()))
                t1 = 0
                for new_path, new_list_T in zip(inputs.paths_to_data, inputs.list_T_per_individual_path):
                # for j in range(nb_paths):

                    new_data = import_data(inputs,mF,new_path, new_list_T)[:, a::2, :] # shape n d t
                    t2 = t1+new_data.shape[-1]

                    result_cut_nP = einsum(nodes_to_gauss(new_data**2, inputs), inputs.W, "N D T, N -> T")
                    cumulated_energy_NP_T = np.zeros((inputs.phys_pod_modes_to_save.shape[0]+1, result_cut_nP.shape[0]))

                    cumulated_energy_NP_T[0, :] = result_cut_nP

                    sfem_par.add_bins(path_bins+f"m{m:03d}/", field=f'POD_{inputs.field}', D=inputs.D, replace=True)
                    cumulated_field = np.zeros(new_data.shape)
                    for i in range(inputs.phys_pod_modes_to_save.shape[0]):
                        phi = get_fourier(sfem_par, I=i+1)[:, a::2, np.array([mF])]
                        cumulated_field += phi*(latents[i, t1:t2]).reshape(1, 1, t2-t1)
                        result_cut_nP = einsum(nodes_to_gauss((new_data-cumulated_field)**2, inputs), inputs.W, "N D T, N -> T")

                        cumulated_energy_NP_T[i+1, :] = result_cut_nP

                    if mF != 0:
                        energy_difference_matrix[:, t1:t2] += cumulated_energy_NP_T/2
                    if mF == 0:
                        energy_difference_matrix[:, t1:t2] += cumulated_energy_NP_T

                    # else:
                    #     energy_difference_matrix = np.concatenate((energy_difference_matrix, cumulated_energy_NP_T), axis=1)
                    t1 = t2
            if inputs.size > 1:
                energy_difference_matrix = inputs.comm_axis.reduce(energy_difference_matrix, root=0)
        if inputs.rank_axis == 0:
            if inputs.size > 1:
                energy_difference_matrix = inputs.comm_fourier.reduce(energy_difference_matrix, root=0)
            if inputs.rank_fourier == 0:
                difference_NP = np.diff(energy_difference_matrix, axis=0)
                location_wrong_cumulation = np.where(np.min(difference_NP, axis=0)>0)[0]
                write_job_output(inputs, f"    In {m}-family: total of {location_wrong_cumulation.sum()} incorrect summation for snapshots")
                np.save(path_out+f"energy_difference_{m}.npy",energy_difference_matrix)


def test_cumulated_energy(inputs, sfem_par, eps_cum_en=1e-5, eps_mean_latents=1e-7):
    path_bins = inputs.complete_output_path+inputs.output_file_name+f"/phys_pod_modes/"
    path_latents = inputs.complete_output_path+inputs.output_file_name
    nb_paths = len(inputs.paths_to_data)
    path_out = inputs.complete_output_path+inputs.output_file_name+f"/post_tests/"

    for m_family in inputs.list_m_families:
        m = m_family.min()%inputs.number_shifts
        write_job_output(inputs,f"      Starting cumulated latents test for {m}-family")

        latents = np.load(path_latents+f"a_phys_m{m}.npy")
        spec = np.load(path_latents+f"spectrum_phys_m{m}.npy")
        
        list_energies = np.zeros(np.sum(inputs.list_T))
        snapshots_energy = np.zeros(np.sum(inputs.list_T))
        diff = np.abs(spec/np.mean(latents**2, axis=1)-1)>eps_mean_latents
        bool_cumulated_energy = True
        write_job_output(inputs, f"    In {m}-family: total of {diff.sum()} incorrect normalization of latents")
        if diff.sum()>0:
            bool_cumulated_energy = False
        for i,mF in enumerate(inputs.list_modes[inputs.rank_fourier::inputs.nb_proc_in_fourier]):
            factor = 1*(mF==0) + 1/2*(mF>0) 
            for a in range(inputs.rank_axis,2,inputs.nb_proc_in_axis):
                if (mF == 0 and a == 1) or (not mF in m_family):
                    continue

                t1 = 0
                # for j in range(nb_paths):
                for new_path, new_list_T in zip(inputs.paths_to_data, inputs.list_T_per_individual_path):
                    new_data = import_data(inputs,mF,new_path, new_list_T)[:, a::2, :] # shape n d t

                    # new_data = import_data(inputs,mF,inputs.paths_to_data[j])[:, a::2, :] # shape n d t
                    t2 = t1+new_data.shape[-1]
                    snapshots_energy = einsum(nodes_to_gauss(new_data**2, inputs), inputs.W, "N D T, N -> T")
                    list_energies[t1:t2] += factor*snapshots_energy
                    t1 = t2
            if inputs.size > 1:
                list_energies = inputs.comm_axis.reduce(list_energies, root=0)
        if inputs.rank_axis == 0:
            if inputs.size > 1:
                list_energies = inputs.comm_fourier.reduce(list_energies, root=0)
            if inputs.rank_fourier == 0:
                difference = np.abs(list_energies.mean()/spec.sum()-1)
                write_job_output(inputs, f"    In {m}-family: POD spectrum gives mean energy {spec.sum()}, VS actual mean energy {list_energies.mean()} ==> relative difference of {difference}.")
                #if difference>eps_cum_en:
                bool_cumulated_energy = difference<eps_cum_en
                
                if bool_cumulated_energy:
                    write_job_output(inputs, f"            Cumulated latents Test successful for {m}-family")
                else:
                    write_job_output(inputs, f"            Cumulated latents Test failed for {m}-family")
                np.save(path_out+f"snapshots_energy{m}.npy",list_energies)

def test_symmetry(inputs, sfem_par, eps_sym=1e-3):
    path_bins = inputs.complete_output_path+inputs.output_file_name+f"/phys_pod_modes/"
    path_out = inputs.complete_output_path+inputs.output_file_name+f"/post_tests/"
    path_latents = inputs.complete_output_path+inputs.output_file_name
    path_symmetry_POD = inputs.complete_output_path+inputs.output_file_name

    for m_family in inputs.list_m_families:
        m = m_family.min()%inputs.number_shifts
        write_job_output(inputs,f"      Starting symmetry test for {m}-family")

        sfem_par.add_bins(path_bins+f"m{m:03d}/", field=f'POD_{inputs.field}', D=inputs.D, replace=True)
        symmetry_POD = np.load(path_symmetry_POD+f"symmetries_phys_m{m}.npy")
        latents = np.load(path_latents+f"a_phys_m{m}.npy")
        bool_latents_sym = True
        Nt = inputs.list_T.sum()
        first_latents = latents[:, :Nt]
        second_latents = latents[:, Nt:2*Nt]
        summed_latents = (np.abs(first_latents - np.real(symmetry_POD.reshape(symmetry_POD.shape[0], 1))*second_latents)).mean(axis=1)/(np.abs(latents)).mean(axis=1)
        diff = summed_latents>eps_sym
        bool_latents_sym = (diff.sum())==0
        write_job_output(inputs, f"    In {m}-family: total of {diff.sum()} incorrect symmetrization of latents")
        
        if bool_latents_sym:
            write_job_output(inputs, f"            Latents Symmetry Test successful for {m}-family")
        else:
            write_job_output(inputs, f"            Latents Symmetry Test failed for {m}-family")
            write_job_output(inputs, str(diff))
        symmetry_matrix = np.zeros(inputs.phys_pod_modes_to_save.shape[0])
        bool_POD_sym = True
        for i in range(inputs.rank, inputs.phys_pod_modes_to_save.shape[0], inputs.size):
            nP = inputs.phys_pod_modes_to_save[i]

            phi = get_fourier(sfem_par, I=nP+1)
            phi = nodes_to_gauss(phi, inputs)
            phi_sym = np.copy(phi)
            apply_3D_mesh_sym(inputs, phi_sym)
            difference = phi-np.real(symmetry_POD[i])*phi_sym
            
            result = FFT_EUCLIDIAN_PROD(difference, difference, inputs.W)
            symmetry_matrix[i] = result
        symmetry_matrix = inputs.comm.reduce(symmetry_matrix, root=0)
        if inputs.rank == 0:
            for i in range(symmetry_matrix.shape[0]):
                res = symmetry_matrix[i]
                if res/2/np.pi > eps_sym:
                    bool_POD_sym = False
                    write_job_output(inputs, str((i, res/2/np.pi)))
            if bool_POD_sym:
                write_job_output(inputs, f"            POD Symmetry Test successful for {m}-family")
            else:
                write_job_output(inputs, f"            POD Symmetry Test failed for {m}-family")
            
        np.save(path_out+f"symmetry_{m}.npy", symmetry_matrix)
