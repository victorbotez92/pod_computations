import os,array,sys,gc
from memory_profiler import profile

import numpy as np
from einops import rearrange

#####################################################
from read_restart_sfemans import get_data_from_suites
from basic_functions import write_job_output, print_memory_usage

sys.path.append('/gpfs/users/botezv/.venv')
from SFEMaNS_env.operators import nodes_to_gauss, gauss_to_nodes
from SFEMaNS_env.read_stb import get_fourier_per_mode
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

def get_data(path_to_suite,field,mesh_type,mF,D,S,T,N,axis,type_float = np.float64):
    N_tot = np.sum(np.asarray(N))
    N_slice=np.cumsum(np.asarray(N)//T)
    data = np.zeros(shape=(T,D,N_tot//T),dtype=type_float)
    if axis == 'c':
        a = 0
    elif axis == 's':
        a = 1
    else:
        raise ValueError(f"error in value of axis (found {axis}, should be c or s)")
    for s in range(S):
        n = N[s]*2*D
            # for a,axis in enumerate(fourier_type):
        path=path_to_suite+"/fourier_{f}_S{s:04d}_F{m:04d}".format(f=field,s=s,m=mF)+mesh_type

        new_data = np.asarray(get_file(path,n),dtype=type_float)
        new_data = rearrange(new_data, '(T d N) -> T N d', T=T, d=D*2)#new_data.reshape(T,len(new_data)//T)
        new_data = new_data[:, :, a::2]
        if s==0:
            data = np.copy(new_data)
        else:
            data = np.hstack((data, new_data))
    return rearrange(data, 'T N d -> T d N')

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
    bool_import_mean_field = (mF == 0 and axis=='c') or ((mF != 0) and (mF in par.list_m_families[0]) and (par.should_mean_field_be_axisymmetric == False))
    if bool_import_mean_field:
        return np.load(par.complete_output_path+f'/mean_field/mF{mF}_{axis}.npy')
    else:
        return 0

################################# MAIN FUNCTION

# @profile
def import_data(inputs,mF,raw_paths_to_data_shift, list_T=[-1]):
    print(list_T, raw_paths_to_data_shift)
    ####################################################### IMPORTING SUCCESSIVELY ALL PATHS TO DEAL WITH AS ONE
    list_data = []

    for num,raw_path_to_data_shift_i_integer in enumerate(raw_paths_to_data_shift):

        if '.shifted' in raw_path_to_data_shift_i_integer:
            path_to_data = raw_path_to_data_shift_i_integer.split('.')[0] #!!!!!!!!!!!!! PB HERE IF THE ORIGINAL DIRECTORY CONTAINS A DOT SOMEWHERE !!!!!!!
            num_angle = int(raw_path_to_data_shift_i_integer.split('_')[-1])
            to_be_shifted = True
        else:
            path_to_data = raw_path_to_data_shift_i_integer
            to_be_shifted = False

        # print_memory_usage(inputs, tag="In import_data ==> Before importing")


        inputs.sfem_par.add_bins(inputs.path_to_suites+path_to_data, inputs.field_name_in_file, D=inputs.D, replace=True, from_gauss=inputs.read_from_gauss)

        raw_new_data = get_fourier_per_mode(inputs.sfem_par, T=list_T[num], mF=mF)


        new_data = np.transpose(raw_new_data, (1,2,0)) #     "T N D -> N D T"
        del raw_new_data
        gc.collect()

        if inputs.read_from_gauss:
            new_data = gauss_to_nodes(new_data, inputs)

        ############################ APPLYING THE TRANSFORMATIONS THAT ARE REQUIRED
        list_data.append(new_data)

    full_data = np.concatenate(list_data, axis=2)

    del list_data, new_data
    gc.collect()

    return full_data
