import os,sys, psutil

from mpi4py import MPI

import numpy as np
from einops import rearrange,einsum

########################################################################
########################################################################
#                   IMPORTING POD_ENV
########################################################################
########################################################################
from read_data import parameters
from basic_functions import write_job_output,indiv_ranks,invert_rank
from functions_to_get_data import import_data

########################################################################
########################################################################
#                   IMPORTING SFEMaNS_ENV
########################################################################
########################################################################
sys.path.append('/gpfs/users/botezv/.venv')
from SFEMaNS_env.SFEMaNS_object import define_mesh, SFEMaNS_par
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


    inputs = parameters(data_file)

########################################################################
########################################################################
#                   VALIDATING DEFINITION OF FIELD IN DATA FILE
########################################################################
########################################################################

    field = inputs.field
    D = inputs.D
    mesh_type = inputs.mesh_type

    if (not D is None) and field in dict_for_fields.keys() and dict_for_fields[field][0] != D:
        raise ValueError(f"You set dimension D = {D} while for {field} it should be {dict_for_fields[field][0]}")
    elif D is None:
        if not field in dict_for_fields.keys():
            raise TypeError(f"field {field} not known, please set manually its dimension in data file")
        else:
            D = dict_for_fields[field][0]
            inputs.D = D

    if (not mesh_type is None) and field in dict_for_fields.keys() and dict_for_fields[field][1] != mesh_type:
        raise ValueError(f"You set mesh_type {mesh_type} while for {field} it should be {dict_for_fields[field][1]}")
    elif mesh_type is None:
        if not field in dict_for_fields.keys():
            raise TypeError(f"field {field} not known, please set manually its mesh_type in data file")
        else:
            mesh_type = dict_for_fields[field][1]
            inputs.mesh_type = mesh_type

    sfem_par = SFEMaNS_par(inputs.path_to_mesh, mesh_type=mesh_type)
    S = sfem_par.S
    inputs.S = S
    inputs.sfem_par = sfem_par


########################################################################
########################################################################
# Defining list_modes
########################################################################
########################################################################

    if inputs.opt_mF[0] == -1:
        list_modes = np.arange(inputs.MF)
    else:
        list_modes = inputs.opt_mF

    inputs.list_modes = list_modes

    if inputs.save_bins_format:
        if not inputs.bins_format in ['fourier', 'phys']:
            raise ValueError(f"In bins_format, you chose {inputs.bins_format}. Please pick fourier or phys")
        if inputs.bins_format == 'phys':
            raise TypeError("selected phys-like output binaries, not available yet in snapshots from SFEMaNS V6")
########################################################################
########################################################################
# CHECKING ASSERTIONS ABOUT DATA-AUGMENTATION AND MANIPULATION
########################################################################
########################################################################

    assert (inputs.is_the_field_to_be_renormalized_by_its_L2_norm and inputs.is_the_field_to_be_renormalized_by_magnetic_energy) == False
    assert not (inputs.should_we_restrain_to_symmetric and inputs.should_we_restrain_to_antisymmetric)
    assert not ((inputs.should_we_restrain_to_symmetric or inputs.should_we_restrain_to_antisymmetric) and inputs.should_we_add_mesh_symmetry)
    
    inputs.renormalize = (inputs.is_the_field_to_be_renormalized_by_magnetic_energy or inputs.is_the_field_to_be_renormalized_by_its_L2_norm)
    inputs.mean_field = inputs.should_we_remove_mean_field
########################################################################
########################################################################
# CREATING ALL RELEVANT PATHS/FOLDERS
########################################################################
########################################################################

    output_file_name = inputs.output_file_name
    output_path = inputs.output_path

    complete_output_path = inputs.path_to_suites + '/' + output_path

    inputs.complete_output_path = complete_output_path
    

    if "D00" in field:
        field_name_in_file = field[5:]
    else:
        field_name_in_file = field

    inputs.field_name_in_file = field_name_in_file

    os.makedirs(complete_output_path,exist_ok=True)
    os.makedirs(complete_output_path+"/"+output_file_name,exist_ok=True)

    os.makedirs(complete_output_path+"/"+output_file_name+"/latents" ,exist_ok=True)

    os.makedirs(complete_output_path+"/"+output_file_name+"/energies" ,exist_ok=True)

    if inputs.should_we_add_mesh_symmetry:
        os.makedirs(complete_output_path+"/"+output_file_name+"/symmetry" ,exist_ok=True)

    if inputs.should_we_save_Fourier_POD:
        os.makedirs(complete_output_path+output_file_name+"/fourier_pod_modes",exist_ok=True)
    if inputs.should_we_save_phys_POD:
        os.makedirs(complete_output_path+output_file_name+"/phys_pod_modes",exist_ok=True)

    if inputs.do_post_tests:
        os.makedirs(complete_output_path+output_file_name+"/post_tests",exist_ok=True)


# os.makedirs(f"{complete_output_path + '/JobLogs_outputs'}",exist_ok=True)
    path_to_job_output = complete_output_path + '/' + output_file_name + '/JobLogs_outputs/'
    os.makedirs(path_to_job_output, exist_ok=True)
    path_to_job_output += inputs.name_job_output
    inputs.path_to_job_output = path_to_job_output
    os.system(f"touch {path_to_job_output}")

    with open(path_to_job_output,'w') as f:
        f.write('')


    
########################################################################
########################################################################
# Create the list containing all paths + rotation symmetries
########################################################################
########################################################################

    if inputs.should_we_combine_with_shifted_data and inputs.number_shifts>1:
        raise Exception(ValueError, "can't have simultaneously 'should_we_combine_with_shifted_data' and 'number_shifts>1'")

################ USING MATRIX REDUCTION 

    if inputs.number_shifts>1:
        list_m_families = []
        for i,mF in enumerate(inputs.list_modes):
        # for i,mF in enumerate(inputs.list_modes[inputs.rank_fourier::inputs.nb_proc_in_fourier]):
            is_present = any(mF in m_family for m_family in list_m_families)
            if is_present:
                continue
            new_family = []
            cur_mF = mF
            while cur_mF < inputs.MF:
                if cur_mF in inputs.list_modes:
                    new_family.append(cur_mF)
                cur_mF += inputs.number_shifts
            cur_mF = mF - inputs.number_shifts

            while cur_mF > -inputs.MF:
                if np.abs(cur_mF) in inputs.list_modes:
                    new_family.append(np.abs(cur_mF))
                cur_mF -= inputs.number_shifts

            # get rid of repeated values
            new_family = list(set(new_family))
            # sort
            new_family = np.sort(np.asarray(new_family))
            list_m_families.append(np.copy(new_family))

    else:
        list_m_families = [inputs.list_modes]
    
    inputs.list_m_families = list_m_families
    if inputs.should_we_save_phys_POD:
        for m_family in list_m_families:
            m = m_family[0]
            os.makedirs(complete_output_path+output_file_name+f"/phys_pod_modes/m{m:03d}",exist_ok=True)

################ MANUALLY SHIFTING DATA

    if inputs.should_we_combine_with_shifted_data:
        nb_paths_to_data = len(inputs.paths_to_data)
        for n in range(len(inputs.shift_angle)):
            for i in range(nb_paths_to_data):
                new_shifts = []
            ##### the ".shifted" can be interpreted within the function "import_data" of "functions_to_get_data"
                local_nb_paths_to_data = len(inputs.paths_to_data[i])
                for j in range(local_nb_paths_to_data):
                    new_shifts.append(inputs.paths_to_data[i][j]+f'.shifted_{n}')
                inputs.paths_to_data.append(new_shifts.copy())

    output_path = inputs.output_path
    output_file_name = inputs.output_file_name


########################################################################
########################################################################
# CREATING MESHES AND RELEVANT WEIGHTS WITH MESH SYMMETRIES
########################################################################
########################################################################

     
    mesh = define_mesh(inputs.path_to_mesh, inputs.mesh_type)
    inputs.jj = mesh.jj
    inputs.ww = mesh.ww

    if inputs.should_we_penalize_divergence or inputs.do_post_tests:
        inputs.dw = mesh.dw
        inputs.R = mesh.R

    W = einsum(mesh.R[mesh.jj], mesh.ww, mesh.rj, 'nw me, nw l_G, l_G me -> l_G me')
    inputs.ww_W = einsum(inputs.ww, inputs.ww, W, "nw1 l_G, nw2 l_G, l_G me -> nw1 nw2 me")
    W = rearrange(W, 'l_G me -> (me l_G)')
    inputs.W = W

        #========== necessary for gauss_to_nodes
    if inputs.save_bins_format or inputs.do_post_tests:
        inputs.l_G = mesh.l_G
        inputs.me = mesh.me
        inputs.rj = mesh.rj
        inputs.nw = mesh.nw
        inputs.R = mesh.R
    
    if inputs.should_we_modify_weights:
        if inputs.directory_scalar_for_weights == '':
            raise ValueError(f"You chose to modify weights but specified wrong path for weights: {inputs.directory_scalar_for_weights}")
        existence = os.path.exists(inputs.directory_scalar_for_weights+'/weight_pod.py')
        if not existence:
            raise ValueError(f"{inputs.directory_scalar_for_weights} does not exist")
        sys.path.append(inputs.directory_scalar_for_weights)
        from weight_pod import axisym_scalar
        to_be_multiplied = axisym_scalar(R, Z)
        if to_be_multiplied.min() <= 0:
            raise ValueError("the funciton axisym_scalar you coded returns negative values, make sure it only returns strictly positive")
        W *= to_be_multiplied
        W /= W.sum()
    WEIGHTS = np.asarray([W for _ in range(D)]).reshape(-1) 
    
    if inputs.should_we_add_mesh_symmetry or inputs.should_we_restrain_to_symmetric or inputs.should_we_restrain_to_antisymmetric:

        R = einsum(mesh.R[mesh.jj], mesh.ww, 'nw me, nw l_G -> l_G me')
        R = rearrange(R, 'l_G me -> (me l_G)')
        Z = einsum(mesh.Z[mesh.jj], mesh.ww, 'nw me, nw l_G -> l_G me')
        Z = rearrange(Z, 'l_G me -> (me l_G)')

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
        inputs.tab_pairs_gauss = tab_pairs

        del R, Z

        mesh.build_tab_sym()
        inputs.tab_pairs_nodes = mesh.tab_sym



########################################################################
#CREATING SYMMETRIZED WEIGHTS
########################################################################
        if D == 3:
            if inputs.type_sym == 'Rpi':
                relative_signs = [1,-1,-1]
            elif inputs.type_sym == 'centro':
                relative_signs = [1,1,-1]
            else:
                raise ValueError(f'type_sym must be either Rpi or centro, the value {inputs.type_sym} is not valid')
            # do the same for centro-symmetry ?
        elif D == 1:
            relative_signs = [1]
        relative_signs = np.asarray(relative_signs)

    else:
        relative_signs = np.ones(D)
    inputs.sym_signs = relative_signs
    # inputs.for_building_symmetrized_weights = for_building_symmetrized_weights
    del mesh

########################################################################
########################################################################
# Initializing MPI
########################################################################
########################################################################

    if not parallelize:
        inputs.nb_proc_in_fourier = 1
        inputs.nb_proc_in_axis = 1
        inputs.nb_proc_in_meridian = 1

    nb_proc_in_fourier = inputs.nb_proc_in_fourier
    nb_proc_in_axis = inputs.nb_proc_in_axis
    nb_proc_in_meridian = inputs.nb_proc_in_meridian

    if nb_proc_in_fourier*nb_proc_in_axis*nb_proc_in_meridian==1 or not parallelize:
        rank = 0
        size = 1
        comm = None
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    assert (size==nb_proc_in_fourier*nb_proc_in_axis*nb_proc_in_meridian or not parallelize)
    if size > len(inputs.list_modes)*2:
        raise IndexError(f"Asked for {size} procs, while only {len(inputs.list_modes)} distinct Fourier modes with axis=2")
    
    inputs.comm = comm
    inputs.rank = rank
    inputs.size = size

    rank_axis,rank_fourier,rank_meridian = indiv_ranks(inputs)
    inputs.rank_axis,inputs.rank_fourier,inputs.rank_meridian = rank_axis,rank_fourier,rank_meridian
    assert rank == invert_rank(inputs.rank_fourier,inputs.rank_axis,inputs.rank_meridian,inputs)

    ########################################################################
    if size > 1:

        gpe_fourier = comm.group.Incl(list(invert_rank(new_rank_fourier,inputs.rank_axis,inputs.rank_meridian,inputs) for new_rank_fourier in np.arange(inputs.nb_proc_in_fourier)))
        comm_fourier = comm.Create_group(gpe_fourier)
        inputs.comm_fourier = comm_fourier

        gpe_axis = comm.group.Incl(list(invert_rank(inputs.rank_fourier,new_rank_axis,inputs.rank_meridian,inputs) for new_rank_axis in np.arange(inputs.nb_proc_in_axis)))
        comm_axis = comm.Create_group(gpe_axis)
        inputs.comm_axis = comm_axis

        gpe_meridian = comm.group.Incl(list(invert_rank(inputs.rank_fourier,inputs.rank_axis,new_rank_meridian,inputs) for new_rank_meridian in np.arange(inputs.nb_proc_in_meridian)))
        comm_meridian = comm.Create_group(gpe_meridian)
        inputs.comm_meridian = comm_meridian


    ########################################################################

    # process = psutil.Process()
    # mem_info = process.memory_info()
    total = psutil.virtual_memory().total #/ (1024 ** 3)
    available_per_process = total/inputs.size
    margin_factor = 3/2*7  #*2 #factor 3/2 of margin
    available_per_process /= margin_factor

    n_gauss = inputs.W.shape[0]
    memory_elementary_tab = n_gauss*2*inputs.D*8 #2 for cos/sin, expressed in bytes (considering np.float64)
    elementary_nb_snapshots = int(np.floor(available_per_process/memory_elementary_tab))

    nb_snapshots = 0
    total_snashots = 0
    list_T_per_individual_path = []
    list_T_per_set = []

    list_T = []

    sample_lists = []
    sample_T = []
    paths_to_data = []

    for new_path in inputs.paths_to_data:
        new_T = import_data(inputs,0,[new_path]).shape[-1]
        list_T.append(new_T)
        total_snashots += new_T
        if nb_snapshots + new_T < elementary_nb_snapshots:
            sample_lists.append(new_path)
            sample_T.append(-1)
            nb_snapshots += new_T

        else:
            if nb_snapshots != 0:
                if not (isinstance(sample_T, list) and sample_T == []):

                    paths_to_data.append(sample_lists)
                    list_T_per_individual_path.append(sample_T)
                    sample_lists = []
                    sample_T = []

            nb_snapshots = 0

            if new_T < elementary_nb_snapshots:
                sample_lists.append(f"{new_path}")
                sample_T.append(-1)
                nb_snapshots += new_T
            else:
                nb_cuts = int(new_T//elementary_nb_snapshots)
                for i in range(nb_cuts):
                    paths_to_data.append([new_path])
                    list_T_per_individual_path.append([np.arange(elementary_nb_snapshots)+i*elementary_nb_snapshots])

                if new_T%elementary_nb_snapshots != 0:
                    sample_T.append(np.arange(new_T%elementary_nb_snapshots)+nb_cuts*elementary_nb_snapshots)
                    sample_lists.append(new_path)
                    nb_snapshots = new_T%elementary_nb_snapshots

    if not (isinstance(sample_T, list) and sample_T == []):

        paths_to_data.append(sample_lists)
        list_T_per_individual_path.append(sample_T)


    inputs.raw_paths_to_data = inputs.paths_to_data
    inputs.list_T = list_T
    
    inputs.paths_to_data = paths_to_data
    inputs.list_T_per_individual_path = list_T_per_individual_path
    inputs.total_snashots = total_snashots

    write_job_output(inputs, f"nb_snapshots: {total_snashots}")
    write_job_output(inputs, f"limited to simultaneous import of {elementary_nb_snapshots} snapshots")


########################################################################
########################################################################
# END INITIALIZATION
########################################################################
########################################################################

    return inputs, sfem_par
