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
def get_data(path_to_suite,field,mesh_type,mF,D,S,T,N,axis,type_float = np.float64):
    N_tot = np.sum(np.array(N))
    N_slice=np.cumsum(np.array(N)//T)
    data = np.zeros(shape=(T,D,N_tot//T),dtype=type_float)
    for s in range(S):
        n = N[s]
        for d in range(D):
            # for a,axis in enumerate(fourier_type):
            if D > 1:
                path=path_to_suite+"/fourier_{f}{i}{ax}_S{s:04d}_F{m:04d}".format(f=field,i=d+1,ax=axis,s=s,m=mF)+mesh_type
            elif D == 1:
                path=path_to_suite+"/fourier_{f}{ax}_S{s:04d}_F{m:04d}".format(f=field,ax=axis,s=s,m=mF)+mesh_type

            new_data = np.array(get_file(path,n),dtype=type_float)
            new_data = new_data.reshape(T,len(new_data)//T)
            if s==0:
                data[:,d,:N_slice[s]]=np.copy(new_data)
            else:
                data[:,d,N_slice[s-1]:N_slice[s]]=np.copy(new_data)
    return data

################################# EFFECTIVELY APPLYING RENORMALIZATION

def apply_renormalization(par,data,path_to_data):
    # if par.is_the_field_to_be_renormalized_by_its_L2_norm:
    renormalize_factor = np.load(par.path_to_suites+path_to_data+'L2_norm.npy')
    renormalize_factor = renormalize_factor[:,np.newaxis]
    # else:
    #     renormalize_factor = 1
    return data/renormalize_factor

################################# IMPORTING MEAN FIELD

def import_mean_field(par, mF, axis):
    if par.should_mean_field_computation_include_mesh_sym:
        char = 'mesh_sym'
    else:
        char = 'no_mesh_sym'                    
    return np.load(par.path_to_suites+f'/mean_field_{char}/mF{mF}_{axis}.npy')

################################# MAIN FUNCTION

def import_data(par,mF,axis,raw_paths_to_data,field_name_in_file,should_we_renormalize=True,building_mean_field=False): # the last parameter is set to False only when creating the normalization coefficients
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
                try:
                    assert tab_snapshots_per_suites[0] == tab_snapshots_per_suites[i]
                except AssertionError:
                    raise IndexError(f"In {raw_path_to_data}: entry files have different number of snapshots {tab_snapshots_per_suites[0]} != {tab_snapshots_per_suites[i]}")

            snapshots_per_suite = int(tab_snapshots_per_suites[0])
            new_data = get_data(par.path_to_suites+path_to_data,
            field_name_in_file,par.mesh_ext,mF,par.D,par.S,snapshots_per_suite,N,axis,type_float = par.type_float)
            new_data = rearrange(new_data,"t d n -> t (d n) ")

        ############################ APPLYING THE TRANSFORMATIONS THAT ARE REQUIRED

        if should_we_renormalize and par.is_the_field_to_be_renormalized_by_its_L2_norm:
            new_data = apply_renormalization(par,new_data,path_to_data)

        if par.should_we_remove_mean_field and not building_mean_field:
            new_data -= import_mean_field(par, mF, axis)

        if par.should_we_restrain_to_symmetric or par.should_we_restrain_to_antisymmetric:
            new_data = rearrange(new_data,"t (d n) -> t d n", d=3)
            coeff = 1
            if par.should_we_restrain_to_antisymmetric:
                coeff *= -1
            if par.type_sym == 'centro':
                if mF%2 == 1:
                    coeff *= -1
                tab_signs = [1, 1, -1]
            if par.type_sym == 'Rpi':
                if axis == 's':
                    coeff *= -1
                tab_signs = [1, -1, -1]
            symmetrized = np.copy(new_data[:, :, par.tab_pairs])
            for d in range(len(new_data[0])):
                new_data[:, d, :] = 1/2*(new_data[:, d, :]+coeff*tab_signs[d]*symmetrized[:, d, :])
            new_data = rearrange(new_data,"t d n -> t (d n) ")

        # if par.should_we_remove_mean_field:
        #     if not shift_component:
        #         new_data -= par.mean_field
        #     else:
        #         new_data -= par.complementary_mean_field
        if num == 0:
            full_data = np.copy(new_data)
        else:
            full_data = np.concatenate((full_data,new_data),axis=0)


    if to_be_shifted and mF != 0: # '.shifted' removed from 'path_to_data' so won't enter this loop in the 'import_data' from below
        if axis == 'c':
            full_data *= par.type_float(np.cos(mF*2*np.pi*par.shift_angle[num_angle]))
            full_data += par.type_float(np.sin(mF*2*np.pi*par.shift_angle[num_angle]))*import_data(par,mF,'s',paths_to_data,field_name_in_file) 
        elif axis == 's':
            full_data *= par.type_float(np.cos(mF*2*np.pi*par.shift_angle[num_angle]))
            full_data += par.type_float(-1*(np.sin(mF*2*np.pi*par.shift_angle[num_angle])))*import_data(par,mF,'c',paths_to_data,field_name_in_file)
    # if par.rank == 0:
    #     write_job_output(par.path_to_job_output,f'full_data: {full_data.dtype}')

    return full_data
