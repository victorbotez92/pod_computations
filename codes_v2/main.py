import sys
from mpi4py import MPI

import shutil


########################################################################
from initialization import init
from compute_renormalizations import renormalization,build_mean_field#, test_normalization
from extract_latents import main_extract_latents
from extract_modes import main_extract_modes, switch_to_bins_format
#from post_test import post_test
from basic_functions import write_job_output

#from post_test import test_normalization,test_divergence,test_cumulated_sum, test_symmetry,test_cumulated_energy
from post_test import main_test
########################################################################

data_file = sys.argv[1]
inputs, sfem_inputs = init(data_file)
write_job_output(inputs,"Initialization done successfully")

########################################################################
########################################################################

#sys.path.append(inputs.path_SFEMaNS_env)
#from read_write_SFEMaNS.read_stb import get_mesh
#from mesh.load_mesh import define_mesh
#from SFEMaNS_object.get_inputs import SFEMaNS_inputs

########################################################################
########################################################################
# job output writing


shutil.copy(data_file, inputs.complete_output_path+"/"+inputs.output_file_name)


########################################################################
########################################################################
################# Compute renormalization coefficients #################
########################################################################
########################################################################

if inputs.renormalize:
    write_job_output(inputs,"=========================================================== BEGINNING RENORMALIZATION")
    renormalization(inputs,inputs.mesh_type)
    if inputs.size != 1:
        inputs.comm.Barrier()
    write_job_output(inputs,"=========================================================== FINISHED RENORMALIZATION")


########################################################################
########################################################################
################# Compute mean-fields ##################################
########################################################################
########################################################################

if inputs.mean_field:
    write_job_output(inputs,"=========================================================== BEGINNING COMPUTATION MEAN FIELD")
    build_mean_field(inputs, inputs.mesh_type, inputs.paths_to_data)
    if inputs.size != 1:
        inputs.comm.Barrier()
    write_job_output(inputs,"=========================================================== FINISHED COMPUTATION MEAN FIELD")


#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################

write_job_output(inputs, "======= FINISHED SNAPSHOTS PRELIMINARY MANIPULATION =======")

if inputs.should_we_save_phys_POD and inputs.rank == 0:
    for m_family in inputs.list_m_families:
        m = m_family.min()%inputs.number_shifts
        write_job_output(inputs,f"{m}-family is {m_family}")
        if m == 0 or (m == inputs.number_shifts//2 and inputs.number_shifts%2 == 0):
            write_job_output(inputs,f'      Not considering crossed correlation matrices for {m}-family')
        else:
            write_job_output(inputs,f'      Considering crossed correlation matrices for {m}-family')

write_job_output(inputs,"The data will be gathered as follows:")
for individual_path_to_data in inputs.paths_to_data:
    write_job_output(inputs,f'  {individual_path_to_data}')

########################################################################
########################################################################
################# Compute time-dependant coefficients ##################
########################################################################
########################################################################

if inputs.should_we_extract_latents:
    write_job_output(inputs,"=========================================================== BEGINNING LATENTS EXTRACTION")
    main_extract_latents(inputs)
    if inputs.size != 1:
        inputs.comm.Barrier()
    write_job_output(inputs,"=========================================================== FINISHED LATENTS EXTRACTION")


########################################################################
########################################################################
################# Compute POD modes ####################################
########################################################################
########################################################################

if inputs.should_we_extract_modes:
    write_job_output(inputs,"=========================================================== BEGINNING MODES EXTRACTION")
    main_extract_modes(inputs)
    if inputs.size != 1:
        inputs.comm.Barrier()
    write_job_output(inputs,"=========================================================== FINISHED MODES EXTRACTION")

if inputs.save_bins_format:
    if inputs.size != 1:
        inputs.comm.Barrier()
    write_job_output(inputs,"=========================================================== BEGINNING SWITCH TO BINS FORMAT")
    switch_to_bins_format(inputs, sfem_inputs)
    if inputs.size != 1:
        inputs.comm.Barrier()
    write_job_output(inputs,"=========================================================== FINISHED SWITCH TO BINS FORMAT")


if inputs.do_post_tests:
    write_job_output(inputs,"=========================================================== BEGINNING POST PROCESSING TESTS")
    main_test(inputs, sfem_inputs)
    if inputs.size != 1:
        inputs.comm.Barrier()
    write_job_output(inputs,"=========================================================== FINISHED POST PROCESSING TESTS")
#   if inputs.size != 1:
#       inputs.comm.Barrier()
#   write_job_output(inputs,"=========================================================== BEGINNING ORTHONORMALITY TEST")
#   test_normalization(inputs, sfem_inputs)
#   if inputs.size != 1:
#       inputs.comm.Barrier()
#   write_job_output(inputs,"=========================================================== FINISHED ORTHONORMALITY TEST")
#
#   write_job_output(inputs,"=========================================================== BEGINNING DIVERGENCE TEST")
#   test_divergence(inputs, sfem_inputs)
#   if inputs.size != 1:
#       inputs.comm.Barrier()
#   write_job_output(inputs,"=========================================================== FINISHED DIVERGENCE TEST")

#   write_job_output(inputs,"=========================================================== BEGINNING POD MODES CUMULATION TEST")
#   test_cumulated_sum(inputs, sfem_inputs)
#   if inputs.size != 1:
#       inputs.comm.Barrier()
#   write_job_output(inputs,"=========================================================== FINISHED POD MODES CUMULATION TEST")
   
#   write_job_output(inputs,"=========================================================== BEGINNING LATENTS CUMULATION TEST")
#   test_cumulated_energy(inputs, sfem_inputs)
#   if inputs.size != 1:
#       inputs.comm.Barrier()
#   write_job_output(inputs,"=========================================================== FINISHED LATENTS CUMULATION TEST")
#
#   if inputs.should_we_add_mesh_symmetry:
#      write_job_output(inputs,"=========================================================== BEGINNING SYMMETRY TEST")
#      test_symmetry(inputs, sfem_inputs)
#      if inputs.size != 1:
#          inputs.comm.Barrier()
#      write_job_output(inputs,"=========================================================== FINISHED SYMMETRY TEST")
   
if inputs.size != 1:
    MPI.Finalize
