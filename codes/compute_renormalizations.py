#!/usr/bin/env python3
import os
from einops import rearrange
#from memory_profiler import profile

#sys.path.append("/ccc/cont003/home/limsi/bousquer/einops")
import numpy as np

###############################
from functions_to_get_data import import_data
from basic_functions import write_job_output,invert_rank
###############################


###############################
import sys
sys.path.append("/gpfs/users/.venv/")
from SFEMaNS_env.operators import gauss_to_nodes, nodes_to_gauss
###############################


########################################################################
########################################################################
########################################################################
########################################################################


def renormalization(par,mesh_type):
    if par.is_the_field_to_be_renormalized_by_its_L2_norm:
        build_L2_renormalization(par, mesh_type)


def build_L2_renormalization(par, mesh_type):
    W = np.hstack([np.fromfile(par.path_to_mesh+f"/{mesh_type}weight_S{s:04d}"+par.mesh_ext) for s in range(par.S)]).reshape(-1)
    WEIGHTS = np.array([W for _ in range(par.D)]).reshape(-1)
    for several_paths_to_data in par.paths_to_data:
        for path_to_data in several_paths_to_data:
            if not ('shifted' in path_to_data):
                if not os.path.exists(par.path_to_suites+'/L2_norms/'+par.output_path+'/'+path_to_data.split('/')[0]+'.npy'): # the calculations are not made if it was already done before
                    once_create_array = False
                    for i,mF in enumerate(par.list_modes[par.rank_fourier::par.nb_proc_in_fourier]):
                    # for mF in range(par.rank_fourier,par.MF,par.nb_proc_in_fourier): # beginning calculations
                        write_job_output(par,f'entering Fourier loop {mF//par.nb_proc_in_fourier+1}/{par.MF//par.nb_proc_in_fourier}')
                        normalize_fourier = (mF == 0) + 1/2*(1-(mF == 0)) # 1/2 if mF > 0, 1 if mF = 0
                        list_axis = ["c","s"]
                        for a in range(par.rank_axis,2,par.nb_proc_in_axis):
                            axis = list_axis[a]
                        ### ==============================================================
                        ### importing data
                        ### ==============================================================
                            new_data = import_data(par,mF,axis,[path_to_data],par.field_name_in_file,should_we_renormalize=False, rm_mean_field=False) # shape t a (d n)
                            write_job_output(par,f'Successfully imported {[path_to_data]}')
                            renormalize_factor = new_data**2*normalize_fourier
                            renormalize_factor = np.sum(WEIGHTS*renormalize_factor,axis=(1,2))

                            if once_create_array == False:
                                new_renormalize_factor = np.copy(renormalize_factor)
                                once_create_array = True
                            else:
                                new_renormalize_factor += renormalize_factor
                        # end for a in list_axis
                    # end for mF in MF
                        ### ==============================================================
                        ### MPI_ALL_REDUCE in axis
                        ### ==============================================================
                    if par.rank_axis != 0:
                        rank_to_send = invert_rank(par.rank_fourier,0,0,par)
                        par.comm.send(new_renormalize_factor,dest=rank_to_send)
                    elif par.rank_axis == 0:
                        for nb_axis in range(1,par.nb_proc_in_axis):
                            rank_recv = invert_rank(par.rank_fourier,nb_axis,0,par)
                            new_renormalize_factor += par.comm.recv(source=rank_recv)
                        write_job_output(par,'Successfully reduced all in axis')

                            ### ==============================================================
                            ### MPI_ALL_REDUCE in Fourier
                            ### ==============================================================
                        if par.rank_fourier != 0:
                            rank_to_send = invert_rank(0,0,0,par)
                            par.comm.send(new_renormalize_factor,dest=rank_to_send)
                        elif par.rank_fourier == 0:
                            for nb_fourier in range(1,par.nb_proc_in_fourier):
                                rank_recv = invert_rank(nb_fourier,0,0,par)
                                new_renormalize_factor += par.comm.recv(source=rank_recv)
                            write_job_output(par,'Successfully reduced all in Fourier')
                            np.save(par.path_to_suites+path_to_data+'L2_norm.npy',np.sqrt(new_renormalize_factor)) # has shape (time)


def build_mean_field(par, mesh_type, paths_to_data):
    for i,mF in enumerate(par.list_modes[par.rank_fourier::par.nb_proc_in_fourier]):
        #if not(mF != 0 and par.should_mean_field_be_axisymmetric):
        bool_compute_mean_field = (mF == 0) or ((mF != 0) and (mF in par.list_m_families[0]) and (par.should_mean_field_be_axisymmetric == False))
        if bool_compute_mean_field:
            MF_output = par.complete_output_path + par.output_file_name + f'/mean_field/'
            if mF == 0: 
                list_axis = ["c"]
            else:
                list_axis = ["c","s"]
            for a in range(par.rank_axis,len(list_axis),par.nb_proc_in_axis):
                axis = list_axis[a]
                ### ==============================================================
                ### Checking mean field calculation has not already been done
                ### ==============================================================
                os.makedirs(MF_output,exist_ok=True)
                if os.path.exists(f'{MF_output}/mF{mF}_{axis}.npy'): # the calculations are not made if it was already done before
                    write_job_output(par,f'mean field already computed for mF = {mF}, axis {axis}')
                else:
                    write_job_output(par,f'computing mean field for mF = {mF}, axis {axis}')
                    once_make_mean_field = False
                    for several_paths_to_data in paths_to_data:
                    ### ==============================================================
                    ### importing data
                    ### ==============================================================
                        # bool_shifted = False
                        # for path in several_paths_to_data:
                        #     if 'shifted' in path:
                        #         bool_shifted = True
                        # if not bool_shifted:
                        new_data = import_data(par,mF,axis,several_paths_to_data,par.field_name_in_file,should_we_renormalize=False,rm_mean_field=False, include_dvg=False) # shape t (d n)
                        write_job_output(par,f'      Successfully imported {several_paths_to_data}')
                        if once_make_mean_field == False:
                            once_make_mean_field = True
                            mean_data = np.sum(new_data, axis = 0)
                            counter = np.shape(new_data)[0]
                        else:
                            mean_data = mean_data + np.sum(new_data, axis = 0)
                            counter += np.shape(new_data)[0]
                    # end for several_paths_to_data in paths_to_data
                    mean_data /= counter
                    ### ==============================================================
                    ### applying symmetry when required
                    ### ==============================================================

                    if par.should_we_add_mesh_symmetry:
                   #     mean_data = rearrange(mean_data, "(d n) -> d n", d=par.D) 
                        mean_data = rearrange(mean_data, "(d n) -> n d 1", d=par.D) 
                        mean_data = nodes_to_gauss(mean_data, par)[:, :, 0].T
                        sym_data = mean_data.copy()
                   #     sym_data = rearrange(sym_data,"(d n) -> d n",d=par.D)
                        if par.type_sym == 'Rpi':
                            for d in range(par.D):
                                d_coeff = ((d == 0)-1/2)*2 # this coeff has value -1 when d > 1 (so for components theta and z) and 1 when d = 1 (so for component r)
                                sym_data[d,:] = d_coeff*sym_data[d,par.tab_pairs]
                            if axis == 's':  # factor -1 when performing rpi-sym on sine components
                                sym_data *= (-1)
                        elif par.type_sym == 'centro':
                            for d in range(par.D):
                                d_coeff = (1/2-(d == 2))*2 # this coeff has value +1 when d = 0,1 (so for compo r & theta) and -1 when d = 2 (so for compo z)
                                sym_data[d,:] = d_coeff*sym_data[d,par.tab_pairs]
                            sym_data *= (-1)**mF# factor -1 when performing centro-sym on even Fourier modes
                        #sym_data = rearrange(sym_data,"d n  -> (d n)")
                        mean_data = (mean_data + sym_data)/2
                        del sym_data

                        mean_data = rearrange(mean_data, "d n -> n d 1", d=par.D)
                        mean_data = gauss_to_nodes(mean_data, par)[:, :, 0]
                        mean_data = rearrange(mean_data, "n d -> (d n)")

                    ### ==============================================================
                    ### saving calculated mean field
                    ### ==============================================================
                    #mean_data = rearrange(mean_data, "(d n) -> n d 1", d=par.D)
                    #mean_data = gauss_to_nodes(mean_data, par, par.W)[:, :, 0]
                    #mean_data = rearrange(mean_data, "n d -> (d n)")
                    print(mean_data.shape)
                    np.save(f'{MF_output}/mF{mF}_{axis}.npy',mean_data) # has shape (d n)
#                    if mF == 0 and axis == 'c':
#                        _, _, WEIGHTS, _ = par.for_building_symmetrized_weights
                        #print("computed mean-field = ", np.sum(nodes_to_gauss(mean_data**2, par)*WEIGHTS))
            par.MF_output = MF_output
                # end if os.path.exist
            # end for a in list_axis
    # end for mF in MF


def test_normalization(par):
    list_pairs = [(n, m) for n in range(par.phys_pod_modes_to_save) for m in range(par.phys_pod_modes_to_save)]
    list_results = np.zeros(10)

