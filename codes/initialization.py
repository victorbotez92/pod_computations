import sys
import os
from mpi4py import MPI

import numpy as np
from einops import rearrange,einsum

########################################################################
from read_data import parameters
from basic_functions import write_job_output,indiv_ranks,invert_rank
########################################################################

list_fields = [('u', 3), ('Tub', 1), ('mu', 1), ('B', 3), ('H', 3), ('Mmu', 1), ('Dsigma', 1)]

dict_for_fields = {
    'u':[3, 'vv'],
    'Tub':[1, 'vv'],
    'mu':[1, 'vv'],
    'B':[3, 'H'],
    'H':[3, 'H'],
    'Mmu':[1, 'H'],
    'Dsigma':[1, 'H'],
}

# list_vv_mesh = ['u','Tub','mu']
# list_H_mesh = ['B','H','Mmu','Dsigma']



def init(data_file, parallelize = True):


    par = parameters(data_file)


########################################################################
########################################################################
#                   IMPORTING SFEMaNS_ENV
########################################################################
########################################################################

    sys.path.append(par.path_SFEMaNS_env)
    from read_write_SFEMaNS.read_stb import get_mesh_gauss
    # from mesh.load_mesh import define_mesh
    # from SFEMaNS_object.get_par import SFEMaNS_par
    from SFEMaNS_object import define_mesh, SFEMaNS_par

########################################################################
########################################################################
#                   VALIDATING DEFINITION OF FIELD IN DATA FILE
########################################################################
########################################################################

    field = par.field
    D = par.D
    mesh_type = par.mesh_type

    if (not D is None) and field in dict_for_fields.keys() and dict_for_fields[field][0] != D:
        raise ValueError(f"You set dimension D = {D} while for {field} it should be {dict_for_fields[field][0]}")
    elif D is None:
        if not field in dict_for_fields.keys():
            raise TypeError(f"field {field} not known, please set manually its dimension in data file")
        else:
            D = dict_for_fields[field][0]
            par.D = D

    if (not mesh_type is None) and field in dict_for_fields.keys() and dict_for_fields[field][1] != mesh_type:
        raise ValueError(f"You set mesh_type {mesh_type} while for {field} it should be {dict_for_fields[field][1]}")
    elif mesh_type is None:
        if not field in dict_for_fields.keys():
            raise TypeError(f"field {field} not known, please set manually its mesh_type in data file")
        else:
            mesh_type = dict_for_fields[field][1]
            par.mesh_type = mesh_type

    sfem_par = SFEMaNS_par(par.path_to_mesh, mesh_type=mesh_type)
    S = sfem_par.S
    par.S = S


########################################################################
########################################################################
# Defining list_modes
########################################################################
########################################################################

    if par.opt_mF[0] == -1:
        list_modes = np.arange(par.MF)
    else:
        list_modes = par.opt_mF

    par.list_modes = list_modes

    if par.save_bins_format:
        if not par.bins_format in ['fourier', 'phys']:
            raise ValueError(f"In bins_format, you chose {par.bins_format}. Please pick fourier or phys")

########################################################################
########################################################################
# CHECKING ASSERTIONS ABOUT DATA-AUGMENTATION AND MANIPULATION
########################################################################
########################################################################

    assert (par.is_the_field_to_be_renormalized_by_its_L2_norm and par.is_the_field_to_be_renormalized_by_magnetic_energy) == False
    assert not (par.should_we_restrain_to_symmetric and par.should_we_restrain_to_antisymmetric)
    assert not ((par.should_we_restrain_to_symmetric or par.should_we_restrain_to_antisymmetric) and par.should_we_add_mesh_symmetry)
    
    par.renormalize = (par.is_the_field_to_be_renormalized_by_magnetic_energy or par.is_the_field_to_be_renormalized_by_its_L2_norm)
    par.mean_field = par.should_we_remove_mean_field
########################################################################
########################################################################
# CREATING ALL RELEVANT PATHS/FOLDERS
########################################################################
########################################################################

    output_file_name = par.output_file_name
    output_path = par.output_path

    complete_output_path = par.path_to_suites + '/' + output_path

    par.complete_output_path = complete_output_path
    

    if "D00" in field:
        field_name_in_file = field[5:]
    else:
        field_name_in_file = field

    par.field_name_in_file = field_name_in_file

    os.makedirs(complete_output_path,exist_ok=True)
    os.makedirs(complete_output_path+"/"+output_file_name,exist_ok=True)

    os.makedirs(complete_output_path+"/"+output_file_name+"/latents" ,exist_ok=True)

    os.makedirs(complete_output_path+"/"+output_file_name+"/energies" ,exist_ok=True)

    if par.should_we_add_mesh_symmetry:
        os.makedirs(complete_output_path+"/"+output_file_name+"/symmetry" ,exist_ok=True)

    if par.should_we_save_Fourier_POD:
        os.makedirs(complete_output_path+output_file_name+"/fourier_pod_modes",exist_ok=True)
    if par.should_we_save_phys_POD:
        os.makedirs(complete_output_path+output_file_name+"/phys_pod_modes",exist_ok=True)


# os.makedirs(f"{complete_output_path + '/JobLogs_outputs'}",exist_ok=True)
    path_to_job_output = complete_output_path + '/' + output_file_name + '/JobLogs_outputs/'
    os.makedirs(path_to_job_output, exist_ok=True)
    path_to_job_output += par.name_job_output
    par.path_to_job_output = path_to_job_output
    os.system(f"touch {path_to_job_output}")

    with open(path_to_job_output,'w') as f:
        f.write('')


########################################################################
########################################################################
# Create the list containing all paths + rotation symmetries
########################################################################
########################################################################

    if par.should_we_combine_with_shifted_data and par.number_shifts>1:
        raise Exception(ValueError, "can't have simultaneously 'should_we_combine_with_shifted_data' and 'number_shifts>1'")

################ USING MATRIX REDUCTION 

    if par.number_shifts>1:
        list_m_families = []
        for i,mF in enumerate(par.list_modes):
        # for i,mF in enumerate(par.list_modes[par.rank_fourier::par.nb_proc_in_fourier]):
            is_present = any(mF in m_family for m_family in list_m_families)
            if is_present:
                continue
            new_family = []
            cur_mF = mF
            while cur_mF < par.MF:
                if cur_mF in par.list_modes:
                    new_family.append(cur_mF)
                cur_mF += par.number_shifts
            cur_mF = mF - par.number_shifts
            while cur_mF > -par.MF:
                new_family.append(np.abs(cur_mF))
                cur_mF -= par.number_shifts

            # get rid of repeated values
            new_family = list(set(new_family))
            # sort
            new_family = np.sort(np.array(new_family))
            list_m_families.append(np.copy(new_family))

    else:
        list_m_families = [np.arange(par.MF)]
        
    par.list_m_families = list_m_families

    if par.should_we_save_phys_POD:
        for m_family in list_m_families:
            m = m_family[0]
            os.makedirs(complete_output_path+output_file_name+f"/phys_pod_modes/m{m:03d}",exist_ok=True)

################ MANUALLY SHIFTING DATA

    if par.should_we_combine_with_shifted_data:
        nb_paths_to_data = len(par.paths_to_data)
        for n in range(len(par.shift_angle)):
            for i in range(nb_paths_to_data):
                new_shifts = []
            ##### the ".shifted" can be interpreted within the function "import_data" of "functions_to_get_data"
                local_nb_paths_to_data = len(par.paths_to_data[i])
                for j in range(local_nb_paths_to_data):
                    new_shifts.append(par.paths_to_data[i][j]+f'.shifted_{n}')
                par.paths_to_data.append(new_shifts.copy())

    output_path = par.output_path
    output_file_name = par.output_file_name


########################################################################
########################################################################
# CREATING MESHES AND RELEVANT WEIGHTS WITH MESH SYMMETRIES
########################################################################
########################################################################

    if not par.read_from_gauss:
        mesh = define_mesh(par.path_to_mesh, par.mesh_type)
        par.jj = mesh.jj
        par.ww = mesh.ww
        
        R = einsum(mesh.R[mesh.jj], mesh.ww, 'nw me, nw l_G -> l_G me')
        R = rearrange(R, 'l_G me -> (me l_G)')
        par.R = R
        
        Z = einsum(mesh.Z[mesh.jj], mesh.ww, 'nw me, nw l_G -> l_G me')
        Z = rearrange(Z, 'l_G me -> (me l_G)')
        par.Z = Z

        W = einsum(mesh.R[mesh.jj], mesh.ww, mesh.rj, 'nw me, nw l_G, l_G me -> l_G me')
        W = rearrange(W, 'l_G me -> (me l_G)')
        par.W = W
     
        del mesh
    
    #R, Z, W = get_mesh_gauss(sfem_par)
    #par.R = R
    #par.Z = Z
    if par.should_we_modify_weights:
        if par.directory_scalar_for_weights == '':
            raise ValueError(f"You chose to modify weights but specified wrong path for weights: {par.directory_scalar_for_weights}")
        existence = os.path.exists(par.directory_scalar_for_weights+'/weight_pod.py')
        if not existence:
            raise ValueError(f"{par.directory_scalar_for_weights} does not exist")
        sys.path.append(par.directory_scalar_for_weights)
        from weight_pod import axisym_scalar
        to_be_multiplied = axisym_scalar(R, Z)
        if to_be_multiplied.min() <= 0:
            raise ValueError("the funciton axisym_scalar you coded returns negative values, make sure it only returns strictly positive")
        W *= to_be_multiplied
        W /= W.sum()
    WEIGHTS = np.array([W for _ in range(D)]).reshape(-1) 

#    if not par.read_from_gauss:
#        mesh = define_mesh(par.path_to_mesh, par.mesh_type)
#        par.jj = mesh.jj
#        par.ww = mesh.ww
#        del mesh
    
    if par.should_we_add_mesh_symmetry or par.should_we_restrain_to_symmetric or par.should_we_restrain_to_antisymmetric:
########################################################################
#IMPORTING LIST_PAIRS
########################################################################
        # ADAPT WHEN USING SEVERAL PROCS IN MERIDIAN
        epsilon_z_0 = 1e-7
        
        partial_sort = np.argsort(R**3+Z**2)
        mask_z_is_not_0 = np.abs(Z[partial_sort])>epsilon_z_0
        
        flip_partial_sort = np.copy(partial_sort)
        restriction_z_is_not_0 = np.zeros(mask_z_is_not_0.sum(), dtype=np.int32)
        
        restriction_z_is_not_0[1::2] = partial_sort[mask_z_is_not_0][::2]
        restriction_z_is_not_0[::2] = partial_sort[mask_z_is_not_0][1::2]
        
        flip_partial_sort[mask_z_is_not_0] = restriction_z_is_not_0
        
        inverse_partial_sort = np.empty(partial_sort.shape[0],dtype=np.int32)
        inverse_partial_sort[partial_sort] = np.arange(partial_sort.shape[0])
        
        tab_pairs = flip_partial_sort[inverse_partial_sort]
        par.tab_pairs = tab_pairs
#        pairs=f"list_pairs_{mesh_type}.npy"
#        list_pairs = np.load(par.path_to_mesh+pairs)
#        tab_pairs = np.empty(2*len(list_pairs),dtype=np.int64)
#        for elm in list_pairs:
#            index,sym_index = elm
#            tab_pairs[index] = int(sym_index)
#            tab_pairs[sym_index] = int(index)
#        par.tab_pairs = tab_pairs
########################################################################
#CREATING SYMMETRIZED WEIGHTS
########################################################################
        if D == 3:
            if par.type_sym == 'Rpi':
                relative_signs = [1,-1,-1]
            elif par.type_sym == 'centro':
                relative_signs = [1,1,-1]
            else:
                raise ValueError(f'type_sym must be either Rpi or centro, the value {par.type_sym} is not valid')
            # do the same for centro-symmetry ?
        elif D == 1:
            relative_signs = [1]

        WEIGHTS_with_symmetry = np.array([relative_signs[d]*W for d in range(D)]).reshape(-1)  
        rows = []
        columns = []
#        for elm in list_pairs:
#            index,sym_index = elm
#            rows.append(index)
#            columns.append(sym_index)
#            rows.append(sym_index)
#            columns.append(index)
        
#        rows = np.array(rows)
#        columns = np.array(columns)

        rows = np.arange(tab_pairs.shape[0])
        columns = np.copy(tab_pairs)
        for_building_symmetrized_weights = (rows,columns,WEIGHTS,WEIGHTS_with_symmetry)

    else:
        for_building_symmetrized_weights = (None,None,WEIGHTS,None)

    par.for_building_symmetrized_weights = for_building_symmetrized_weights

########################################################################
########################################################################
# Initializing MPI
########################################################################
########################################################################

    nb_proc_in_fourier = par.nb_proc_in_fourier
    nb_proc_in_axis = par.nb_proc_in_axis
    nb_proc_in_meridian = par.nb_proc_in_meridian

    if nb_proc_in_fourier*nb_proc_in_axis*nb_proc_in_meridian==1 or not parallelize:
        rank = 0
        size = 1
        comm = None
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    assert (size==nb_proc_in_fourier*nb_proc_in_axis*nb_proc_in_meridian or not parallelize)

    par.comm = comm
    par.rank = rank
    par.size = size

    rank_axis,rank_fourier,rank_meridian = indiv_ranks(par)
    par.rank_axis,par.rank_fourier,par.rank_meridian = rank_axis,rank_fourier,rank_meridian
    assert rank == invert_rank(par.rank_fourier,par.rank_axis,par.rank_meridian,par)

    ########################################################################
    if size > 1:

        gpe_fourier = comm.group.Incl(list(invert_rank(new_rank_fourier,par.rank_axis,par.rank_meridian,par) for new_rank_fourier in np.arange(par.nb_proc_in_fourier)))
        comm_fourier = comm.Create_group(gpe_fourier)
        par.comm_fourier = comm_fourier

        gpe_axis = comm.group.Incl(list(invert_rank(par.rank_fourier,new_rank_axis,par.rank_meridian,par) for new_rank_axis in np.arange(par.nb_proc_in_axis)))
        comm_axis = comm.Create_group(gpe_axis)
        par.comm_axis = comm_axis

        gpe_meridian = comm.group.Incl(list(invert_rank(par.rank_fourier,par.rank_axis,new_rank_meridian,par) for new_rank_meridian in np.arange(par.nb_proc_in_meridian)))
        comm_meridian = comm.Create_group(gpe_meridian)
        par.comm_meridian = comm_meridian


########################################################################
########################################################################
# END INITIALIZATION
########################################################################
########################################################################

    return par, sfem_par
