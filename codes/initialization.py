import sys
import os
from mpi4py import MPI

import numpy as np


########################################################################
from read_data import global_parameters
from compute_renormalizations import renormalization
from extract_latents import main_extract_latents
from extract_modes import main_extract_modes
from basic_functions import write_job_output,indiv_ranks,invert_rank
########################################################################

data_file = sys.argv[1]

########################################################################
########################################################################

list_ints = ['D','S','MF','nb_proc_in_fourier','nb_proc_in_axis','nb_proc_in_meridian']
list_several_ints = ['fourier_pod_modes_to_save','phys_pod_modes_to_save']
list_floats = []
list_several_floats = ['shift_angle']
list_bools = ['READ_FROM_SUITE','is_the_field_to_be_renormalized_by_magnetic_energy',
                'is_the_field_to_be_renormalized_by_its_L2_norm','should_we_save_Fourier_POD','should_we_save_phys_POD',
                'should_we_save_phys_correlation','should_we_extract_latents','should_we_extract_modes',
                'should_we_add_mesh_symmetry','should_we_combine_with_shifted_data',
                'should_we_save_all_fourier_pod_modes','should_we_save_all_phys_pod_modes']
list_chars = ['mesh_ext','path_to_mesh','directory_pairs','directory_codes','field',
              'path_to_suites','name_job_output','output_path','output_file_name']
list_several_chars = []
list_several_list_chars = ['paths_to_data']


list_vv_mesh = ['u','Tub']
list_H_mesh = ['B','H','Mmu','Dsigma']

########################################################################
########################################################################

par = global_parameters(data_file,list_ints,list_several_ints,list_floats,list_several_floats,list_bools,list_chars,list_several_chars,list_several_list_chars)

READ_FROM_SUITE = par.READ_FROM_SUITE

D = par.D
S = par.S
MF = par.MF

should_we_extract_latents = par.should_we_extract_latents
should_we_extract_modes = par.should_we_extract_modes
should_we_save_Fourier_POD = par.should_we_save_Fourier_POD
should_we_save_phys_POD = par.should_we_save_phys_POD
should_we_save_all_fourier_pod_modes = par.should_we_save_all_fourier_pod_modes
should_we_save_all_phys_pod_modes = par.should_we_save_all_phys_pod_modes

should_we_save_phys_correlation = par.should_we_save_phys_correlation

field = par.field

is_the_field_to_be_renormalized_by_magnetic_energy = par.is_the_field_to_be_renormalized_by_magnetic_energy
is_the_field_to_be_renormalized_by_its_L2_norm = par.is_the_field_to_be_renormalized_by_its_L2_norm
renormalize = (is_the_field_to_be_renormalized_by_magnetic_energy or is_the_field_to_be_renormalized_by_its_L2_norm)

should_we_add_mesh_symmetry = par.should_we_add_mesh_symmetry
should_we_combine_with_shifted_data = par.should_we_combine_with_shifted_data
shift_angle = par.shift_angle


mesh_ext = par.mesh_ext
path_to_mesh = par.path_to_mesh
directory_pairs = par.directory_pairs
directory_codes = par.directory_codes

path_to_suites = par.path_to_suites

########################################################################
########################################################################
# Create the list containing all paths + rotation symmetries
########################################################################
########################################################################

if should_we_combine_with_shifted_data:
    nb_paths_to_data = len(par.paths_to_data)
    for n in range(len(shift_angle)):
        for i in range(nb_paths_to_data):
            new_shifts = []
        ##### the ".shifted" can be interpreted within the function "import_data" of "functions_to_get_data"
            local_nb_paths_to_data = len(par.paths_to_data[i])
            for j in range(local_nb_paths_to_data):
                new_shifts.append(par.paths_to_data[i][j]+f'.shifted_{n}')
            par.paths_to_data.append(new_shifts.copy())

paths_to_data = par.paths_to_data

output_path = par.output_path
output_file_name = par.output_file_name


########################################################################
########################################################################
# Mesh parameters + mesh symmetry pairs
########################################################################
########################################################################


if field in list_vv_mesh:
    mesh_type = 'vv'
elif field in list_H_mesh:
    mesh_type = 'H'

par.mesh_type = mesh_type 


assert (is_the_field_to_be_renormalized_by_its_L2_norm and is_the_field_to_be_renormalized_by_magnetic_energy) == False
assert (field in list_vv_mesh) or (field in list_H_mesh)


if should_we_add_mesh_symmetry:
    pairs=f"list_pairs_{mesh_type}.npy"
    list_pairs = np.load(directory_pairs+pairs)
    tab_pairs = np.empty(2*len(list_pairs),dtype=np.int32)
    for elm in list_pairs:
        index,sym_index = elm
        tab_pairs[index] = int(sym_index)
        tab_pairs[sym_index] = int(index)
    par.tab_pairs = tab_pairs

########################################################################
########################################################################
# Initializing MPI
########################################################################
########################################################################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


nb_proc_in_fourier = par.nb_proc_in_fourier
nb_proc_in_axis = par.nb_proc_in_axis
nb_proc_in_meridian = par.nb_proc_in_meridian

assert (size==nb_proc_in_fourier*nb_proc_in_axis*nb_proc_in_meridian)

par.comm = comm
par.rank = rank
par.size = size

rank_axis,rank_fourier,rank_meridian = indiv_ranks(par)
par.rank_axis,par.rank_fourier,par.rank_meridian = rank_axis,rank_fourier,rank_meridian
assert rank == invert_rank(par.rank_fourier,par.rank_axis,par.rank_meridian,par)

########################################################################
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
# Build the necessary sparse matrices if introducing mesh symmetry
########################################################################
########################################################################

W = np.hstack([np.fromfile(path_to_mesh+f"/{mesh_type}weight_S{s:04d}"+mesh_ext) for s in range(rank_meridian,S,nb_proc_in_meridian)]).reshape(-1)
WEIGHTS = np.array([W for _ in range(D)]).reshape(-1) 

if should_we_add_mesh_symmetry:
    # ADAPT WHEN USING SEVERAL PROCS IN MERIDIAN
    if D == 3:
        relative_signs = [1,-1,-1] #cf Rpi-symmetry for the three components
        # do the same for centro-symmetry ?
    elif D == 1:
        relative_signs = [1]

    WEIGHTS_with_symmetry = np.array([relative_signs[d]*W for d in range(D)]).reshape(-1)  
    rows = []
    columns = []
    for elm in list_pairs:
        index,sym_index = elm
        rows.append(index)
        columns.append(sym_index)
        rows.append(sym_index)
        columns.append(index)
    rows = np.array(rows)
    columns = np.array(columns)

    for_building_symmetrized_weights = (rows,columns,WEIGHTS,WEIGHTS_with_symmetry)

else:
    for_building_symmetrized_weights = (None,None,WEIGHTS,None)

par.for_building_symmetrized_weights = for_building_symmetrized_weights

########################################################################
########################################################################
########################################################################
########################################################################

name_job_output = par.name_job_output

complete_output_path = path_to_suites + '/' + output_path
path_to_job_output = directory_codes + '/JobLogs_outputs/' + name_job_output

par.complete_output_path = complete_output_path
par.path_to_job_output = path_to_job_output

os.system(f"touch {directory_codes + '/JobLogs_outputs'}")
os.system(f"touch {path_to_job_output}")

if rank == 0:
    with open(path_to_job_output,'w') as f:
        f.write('')

    write_job_output(path_to_job_output,"Initialization done successfully")
    write_job_output(path_to_job_output,"The data will be gathered as follows:")
    write_job_output(path_to_job_output,str(par.paths_to_data))

########################################################################
########################################################################
################# Make sure all folders are created ####################
########################################################################
########################################################################

if "D00" in field:
    field_name_in_file = field[5:]
else:
    field_name_in_file = field

par.field_name_in_file = field_name_in_file

os.makedirs(complete_output_path,exist_ok=True)
os.makedirs(complete_output_path+"/"+output_file_name,exist_ok=True)

os.makedirs(complete_output_path+"/"+output_file_name+"/latents" ,exist_ok=True)

os.makedirs(complete_output_path+"/"+output_file_name+"/energies" ,exist_ok=True)

if should_we_add_mesh_symmetry:
    os.makedirs(complete_output_path+"/"+output_file_name+"/symmetry" ,exist_ok=True)

if should_we_save_Fourier_POD:
    os.makedirs(complete_output_path+output_file_name+"/fourier_pod_modes",exist_ok=True)
if should_we_save_phys_POD:
    os.makedirs(complete_output_path+output_file_name+"/phys_pod_modes",exist_ok=True)

########################################################################
########################################################################
################# Compute renormalization coefficients #################
########################################################################
########################################################################

if renormalize:
    if rank == 0:
        write_job_output(path_to_job_output,"=========================================================== BEGINNING RENORMALIZATION")
    renormalization(par,mesh_type)
    if rank == 0:
        write_job_output(path_to_job_output,"=========================================================== FINISHED RENORMALIZATION")
        for i in range(1,size):
            comm.send(None,dest=i)
    else:
        confirmation = comm.recv(source=0)

########################################################################
########################################################################
################# Compute time-dependant coefficients ##################
########################################################################
########################################################################

if should_we_extract_latents:
    if rank == 0:
        write_job_output(path_to_job_output,"=========================================================== BEGINNING LATENTS EXTRACTION")
    main_extract_latents(par)
    if rank == 0:
        write_job_output(path_to_job_output,"=========================================================== FINISHED LATENTS EXTRACTION")
        for i in range(1,size):
            comm.send(None,dest=i)
    else:
        confirmation = comm.recv(source=0)

########################################################################
########################################################################
################# Compute POD modes ####################################
########################################################################
########################################################################

if should_we_extract_modes:
    if rank == 0:
        write_job_output(path_to_job_output,"=========================================================== BEGINNING MODES EXTRACTION")
    main_extract_modes(par)
    if rank == 0:
        write_job_output(path_to_job_output,"=========================================================== FINISHED MODES EXTRACTION")

MPI.Finalize