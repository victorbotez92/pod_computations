#!/usr/bin/env python3
import os

#from memory_profiler import profile

#sys.path.append("/ccc/cont003/home/limsi/bousquer/einops")
import numpy as np

###############################
from functions_to_get_data import import_data
from basic_functions import write_job_output,invert_rank
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
                    for mF in range(par.rank_fourier,par.MF,par.nb_proc_in_fourier): # beginning calculations
                        if par.rank == 0:
                            write_job_output(par.path_to_job_output,f'entering Fourier loop {mF//par.nb_proc_in_fourier+1}/{par.MF//par.nb_proc_in_fourier}')
                        normalize_fourier = (mF == 0) + 1/2*(1-(mF == 0)) # 1/2 if mF > 0, 1 if mF = 0
                        list_axis = ["c","s"]
                        for a in range(par.rank_axis,2,par.nb_proc_in_axis):
                            axis = list_axis[a]
                        ### ==============================================================
                        ### importing data
                        ### ==============================================================
                            new_data = import_data(par,mF,axis,[path_to_data],par.field_name_in_file,should_we_renormalize=False) # shape t a (d n)
                            if par.rank == 0:
                                write_job_output(par.path_to_job_output,f'Successfully imported {[path_to_data]}')
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
                        if par.rank == 0:
                            write_job_output(par.path_to_job_output,'Successfully reduced all in axis')

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
                            if par.rank == 0:
                                write_job_output(par.path_to_job_output,'Successfully reduced all in Fourier')
                            np.save(par.path_to_suites+path_to_data+'L2_norm.npy',np.sqrt(new_renormalize_factor)) # has shape (time)