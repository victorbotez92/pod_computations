import sys
from mpi4py import MPI

import shutil


########################################################################
from initialization import init
from compute_renormalizations import renormalization,build_mean_field
from extract_latents import main_extract_latents
from extract_modes import main_extract_modes, switch_to_bins_format
#from post_test import post_test
from basic_functions import write_job_output
########################################################################

data_file = sys.argv[1]
par, sfem_par = init(data_file)
if par.rank == 0:
    write_job_output(par.path_to_job_output,"Initialization done successfully")

########################################################################
########################################################################

#sys.path.append(par.path_SFEMaNS_env)
#from read_write_SFEMaNS.read_stb import get_mesh
#from mesh.load_mesh import define_mesh
#from SFEMaNS_object.get_par import SFEMaNS_par

########################################################################
########################################################################
# job output writing


shutil.copy(data_file, par.complete_output_path+"/"+par.output_file_name)


########################################################################
########################################################################
################# Compute renormalization coefficients #################
########################################################################
########################################################################

if par.renormalize:
    if par.rank == 0:
        write_job_output(par.path_to_job_output,"=========================================================== BEGINNING RENORMALIZATION")
    renormalization(par,par.mesh_type)
    if par.size != 1:
        par.comm.Barrier()
    if par.rank == 0:
        write_job_output(par.path_to_job_output,"=========================================================== FINISHED RENORMALIZATION")


########################################################################
########################################################################
################# Compute mean-fields ##################################
########################################################################
########################################################################

if par.mean_field:
    if par.rank == 0:
        write_job_output(par.path_to_job_output,"=========================================================== BEGINNING COMPUTATION MEAN FIELD")
    build_mean_field(par, par.mesh_type, par.paths_to_data)
    if par.size != 1:
        par.comm.Barrier()
    if par.rank == 0:
        write_job_output(par.path_to_job_output,"=========================================================== FINISHED COMPUTATION MEAN FIELD")


#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################

if par.rank == 0:
    write_job_output(par.path_to_job_output, "======= FINISHED SNAPSHOTS PRELIMINARY MANIPULATION =======")

if par.should_we_save_phys_POD and par.rank == 0:
    for m_family in par.list_m_families:
        m = m_family[0]
        write_job_output(par.path_to_job_output,f"{m}-family is {m_family}")
        if m == 0 or (m == par.number_shifts//2 and par.number_shifts%2 == 0):
            write_job_output(par.path_to_job_output,f'      Not considering crossed correlation matrices for {m}-family')
        else:
            write_job_output(par.path_to_job_output,f'      Considering crossed correlation matrices for {m}-family')

if par.rank == 0:
    write_job_output(par.path_to_job_output,"The data will be gathered as follows:")
    for individual_path_to_data in par.paths_to_data:
        write_job_output(par.path_to_job_output,f'  {individual_path_to_data}')

########################################################################
########################################################################
################# Compute time-dependant coefficients ##################
########################################################################
########################################################################

if par.should_we_extract_latents:
    if par.rank == 0:
        write_job_output(par.path_to_job_output,"=========================================================== BEGINNING LATENTS EXTRACTION")
    main_extract_latents(par)
    if par.size != 1:
        par.comm.Barrier()
    if par.rank == 0:
        write_job_output(par.path_to_job_output,"=========================================================== FINISHED LATENTS EXTRACTION")


########################################################################
########################################################################
################# Compute POD modes ####################################
########################################################################
########################################################################

if par.should_we_extract_modes:
    if par.rank == 0:
        write_job_output(par.path_to_job_output,"=========================================================== BEGINNING MODES EXTRACTION")
    main_extract_modes(par)
    if par.size != 1:
        par.comm.Barrier()
    if par.rank == 0:
        write_job_output(par.path_to_job_output,"=========================================================== FINISHED MODES EXTRACTION")

if par.save_bins_format:
    if par.size != 1:
        par.comm.Barrier()
    if par.rank == 0:
        write_job_output(par.path_to_job_output,"=========================================================== BEGINNING SWITCH TO BINS FORMAT")
    switch_to_bins_format(par, sfem_par)
    if par.size != 1:
        par.comm.Barrier()
    if par.rank == 0:
        write_job_output(par.path_to_job_output,"=========================================================== FINISHED SWITCH TO BINS FORMAT")



if par.size != 1:
    MPI.Finalize
