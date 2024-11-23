import os,array

import numpy as np
from einops import rearrange

from read_restart_sfemans import get_data_from_suites

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
    data = np.zeros(shape=(T,D,2,N_tot//T))
    for s in range(S):
        n = N[s]
        for d in range(D):
            for a,axis in enumerate(fourier_type):
                if D > 1:
                    path=path_to_suite+"/fourier_{f}{i}{ax}_S{s:04d}_F{m:04d}".format(f=field,i=d+1,ax=axis,s=s,m=mF)+mesh_type
                elif D == 1:
                    path=path_to_suite+"/fourier_{f}{ax}_S{s:04d}_F{m:04d}".format(f=field,ax=axis,s=s,m=mF)+mesh_type

                new_data = np.array(get_file(path,n))
                new_data = new_data.reshape(T,len(new_data)//T)
                if s==0:
                    #print(np.shape(new_data),np.shape(data[:,d,a,:N_slice[s]]),np.shape(data))
                    #print(N_slice,N)
                    data[:,d,a,:N_slice[s]]=np.copy(new_data)
                else:
                    data[:,d,a,N_slice[s-1]:N_slice[s]]=np.copy(new_data)
                
    return rearrange(data,"t d a n -> t a (d n) ")

def import_data(par,mF,axis,path_to_data,field_name_in_file):
    size_mesh = [len(np.fromfile(par.path_to_mesh+f"/{par.mesh_type}rr_S{s:04d}"+par.mesh_ext)) for s in range(par.S)]
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
    return new_data


 