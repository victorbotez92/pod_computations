import os,array

import numpy as np
from einops import rearrange

#####################################################
from read_restart_sfemans import get_data_from_suites
from basic_functions import write_job_output
#####################################################

def get_size(path):
    with open(path, 'rb') as fin:
        n = os.fstat(fin.fileno()).st_size // 8
    return n

def get_file(path,n):
    data = array.array('d')
    with open(path, 'rb') as fin:
        data.fromfile(fin, n)
    return data

# @jit(forceobj=True)
def get_data(path_to_suite,field,mesh_type,mF,D,S,T,N,fourier_type = ['c','s']):
    N_tot = np.sum(np.array(N))
    N_slice=np.cumsum(np.array(N)//T)
    data = np.zeros(shape=(T,D,2,N_tot//T),dtype=np.float32)
    for s in range(S):
        n = N[s]
        for d in range(D):
            for a,axis in enumerate(fourier_type):
                if D > 1:
                    path=path_to_suite+"/fourier_{f}{i}{ax}_S{s:04d}_F{m:04d}".format(f=field,i=d+1,ax=axis,s=s,m=mF)+mesh_type
                elif D == 1:
                    path=path_to_suite+"/fourier_{f}{ax}_S{s:04d}_F{m:04d}".format(f=field,ax=axis,s=s,m=mF)+mesh_type

                new_data = np.array(get_file(path,n),dtype=np.float32)
                new_data = new_data.reshape(T,len(new_data)//T)
                if s==0:
                    data[:,d,a,:N_slice[s]]=np.copy(new_data)
                else:
                    data[:,d,a,N_slice[s-1]:N_slice[s]]=np.copy(new_data)
    return data

################################# EFFECTIVELY APPLYING RENORMALIZATION

def apply_renormalization(par,data,path_to_data):
    if par.is_the_field_to_be_renormalized_by_its_L2_norm:
        renormalize_factor = np.load(par.path_to_suites+path_to_data+'L2_norm.npy')
        renormalize_factor = renormalize_factor[:,np.newaxis]
    else:
        renormalize_factor = 1
    return data[:,0]/renormalize_factor

################################# MAIN FUNCTION

def import_data(par,mF,axis,raw_paths_to_data,field_name_in_file,should_we_renormalize=True): # the last parameter is set to False only when creating the normalization coefficients
    size_mesh = [len(np.fromfile(par.path_to_mesh+f"/{par.mesh_type}rr_S{s:04d}"+par.mesh_ext)) for s in range(par.S)]
    ####################################################### IMPORTING SUCCESSIVELY ALL PATHS TO DEAL WITH AS ONE
    paths_to_data = []
    for num,raw_path_to_data in enumerate(raw_paths_to_data):
        if '.shifted' in raw_path_to_data:
            path_to_data = raw_path_to_data.split('.')[0] #!!!!!!!!!!!!! PB HERE IF THE ORIGINAL DIRECTORY CONTAINS A DOT SOMEWHERE !!!!!!!
            num_angle = int(raw_path_to_data.split('_')[-1])
            to_be_shifted = True
        else:
            path_to_data = raw_path_to_data
            to_be_shifted = False

        paths_to_data.append(path_to_data)

        if par.READ_FROM_SUITE:
            new_data = get_data_from_suites(par.path_to_suites,
                            par.path_to_mesh,
                            mF,
                            par.S,
                            field_name_in_file=par.field,
                            record_stack_lenght=7,
                            get_gauss_points=True,
                            stack_domains=True)
            if axis == "c":
                new_data = new_data[...,::2]
            else :
                new_data = new_data[...,1::2]
            new_data = rearrange(new_data,"t n d -> t (d n)")[:,None,:]
        else:
            if par.D > 1:
                N = [get_size(par.path_to_suites+path_to_data+f"/fourier_{field_name_in_file}1c_S{s:04d}_F0000"+par.mesh_ext) for s in range(par.S) ]# get file size for fast import
            elif par.D == 1:
                N = [get_size(par.path_to_suites+path_to_data+f"/fourier_{field_name_in_file}c_S{s:04d}_F0000"+par.mesh_ext) for s in range(par.S) ]# get file size for fast import
            tab_snapshots_per_suites = [N[s]/size_mesh[s] for s in range(par.S)]
            for i in range(1,len(tab_snapshots_per_suites)):
                assert tab_snapshots_per_suites[0] == tab_snapshots_per_suites[i]
            snapshots_per_suite = int(tab_snapshots_per_suites[0])
            new_data = get_data(par.path_to_suites+path_to_data,
            field_name_in_file,par.mesh_ext,mF,par.D,par.S,snapshots_per_suite,N,fourier_type=[axis])
            # if par.rank == 0:
            #     write_job_output(par.path_to_job_output,f'get data: {new_data.dtype}')
            new_data = rearrange(new_data,"t d a n -> t a (d n) ")
            # if par.rank == 0:
            #     write_job_output(par.path_to_job_output,f'rearrange: {new_data.dtype}')

        ############################ APPLYING THE TRANSFORMATIONS THAT ARE REQUIRED

        if should_we_renormalize:
            new_data = apply_renormalization(par,new_data,path_to_data)
        if par.should_we_remove_custom_field:
            new_data = new_data-par.fct_for_custom_field(mF,par.R,par.Z)[np.newaxis,:]

        if num == 0:
            full_data = np.copy(new_data)
        else:
            full_data = np.concatenate((full_data,new_data),axis=0)


    if to_be_shifted and mF != 0: # '.shifted' removed from 'path_to_data' so won't enter this loop in the 'import_data' from below
        if axis == 'c':
            full_data *= np.float32(np.cos(mF*2*np.pi*par.shift_angle[num_angle]))
            full_data += np.float32(np.sin(mF*2*np.pi*par.shift_angle[num_angle]))*import_data(par,mF,'s',paths_to_data,field_name_in_file) 
        elif axis == 's':
            full_data *= np.float32(np.cos(mF*2*np.pi*par.shift_angle[num_angle]))
            full_data += np.float32(-1*(np.sin(mF*2*np.pi*par.shift_angle[num_angle])))*import_data(par,mF,'c',paths_to_data,field_name_in_file)
    # if par.rank == 0:
    #     write_job_output(par.path_to_job_output,f'full_data: {full_data.dtype}')

    return full_data