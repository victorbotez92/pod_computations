import numpy as np

#######################################################
#######################################################
#######################################################
#######################################################
#######################################################

list_ints = ['D','S','MF','nb_proc_in_fourier','nb_proc_in_axis','nb_proc_in_meridian','nb_bits','number_shifts']
list_several_ints = ['fourier_pod_modes_to_save','phys_pod_modes_to_save','opt_mF']
list_floats = ['penal_div']
list_several_floats = ['shift_angle']
list_bools = ['READ_FROM_SUITE','is_the_field_to_be_renormalized_by_magnetic_energy',
                'is_the_field_to_be_renormalized_by_its_L2_norm','should_we_save_Fourier_POD','should_we_save_phys_POD',
                'should_we_save_phys_correlation','should_we_extract_latents','should_we_extract_modes',
                'should_we_add_mesh_symmetry','should_we_combine_with_shifted_data',
                'should_we_save_all_fourier_pod_modes','should_we_save_all_phys_pod_modes',
                'should_we_remove_mean_field',#'should_mean_field_computation_include_mesh_sym',
                'should_we_restrain_to_symmetric','should_we_restrain_to_antisymmetric','save_bins_format','should_we_do_all_post_tests',
                'should_we_modify_weights','should_we_penalize_divergence','should_mean_field_be_axisymmetric',
'read_from_gauss']
list_chars = ['mesh_ext','path_to_mesh','field',
              'path_to_suites','name_job_output','output_path','output_file_name','type_sym',
              'bins_format','mesh_type','directory_scalar_for_weights']

list_several_chars = []
list_several_list_chars = ['paths_to_data']
list_fcts = []



list_types = [list_ints, list_several_ints, list_floats, list_several_floats, list_bools, 
              list_chars, list_several_chars, list_several_list_chars, list_fcts]


#######################################################
#######################################################
#######################################################
#######################################################
#######################################################

def find_string(lines, str_test):
    test = False
    if not ((str_test in list_ints) or (str_test in list_several_ints) or (str_test in list_floats) or (str_test in list_several_floats) or (str_test in list_ints)
            or (str_test in list_bools) or (str_test in list_chars) or (str_test in list_several_chars) or (str_test in list_several_list_chars)
            or (str_test in list_fcts)):
        raise NameError(f"parameter {str_test} from data file not defined in read_data.py")
    
    for i,line in enumerate(lines):
        line = line.split('\n')[0]
        if '===' in line:
            if str_test == line.split('===')[1]:
                # new_param == line[i+1].split('\n')[0]
                test = True

                if str_test in list_several_chars:
                    new_param = lines[i+1].split('\n')[0]
                    new_param = new_param.split(',')
                    list_new_params = []
                    for new_char in new_param:
                        list_new_params.append(new_char)
                    output = list_new_params
                    # globals()[str_test] = list_new_params

                elif str_test in list_several_list_chars:
                    list_new_params = []
                    j = i+1
                    new_param = lines[j].split('\n')[0]
                    while len(new_param) > 0:
                        list_new_params.append(new_param.split(','))
                        j += 1
                        new_param = lines[j].split('\n')[0]
                    output = list_new_params
                    # globals()[str_test] = list_new_params

                elif str_test in list_several_ints:
                    new_param = lines[i+1].split('\n')[0]
                    new_param = new_param.split(',')
                    list_new_params = np.empty(len(new_param),dtype=int)
                    for k,num in enumerate(new_param):
                        list_new_params[k] = int(num)
                    list_new_params = np.array(list_new_params,dtype=int)
                    output = list_new_params
                    globals()[str_test] = list_new_params

                elif str_test in list_several_floats:
                    new_param = lines[i+1].split('\n')[0]
                    new_param = new_param.split(',')
                    list_new_params = np.empty(len(new_param),dtype=float)
                    for k,num in enumerate(new_param):
                        list_new_params[k] = float(num)
                    list_new_params = np.array(list_new_params,dtype=float)
                    output = list_new_params
                    globals()[str_test] = list_new_params
                    

                # elif new_name in list_fcts:
                #     if new_name == "fct_for_custom_field":
                #         new_param = raw_lines[i+1].split('\n')[0]
                #         new_param = new_param.split(',')[0]

                #         fct_for_custom_field = globals()[new_param]
                        # globals()[new_name] = globals()[new_param]
                elif str_test in list_ints:
                    new_param = lines[i+1].split('\n')[0]
                    output = int(new_param)
                    globals()[str_test] = int(new_param)

                elif str_test in list_floats:
                    new_param = lines[i+1].split('\n')[0]
                    output = float(new_param)
                    globals()[str_test] = float(new_param)

                elif str_test in list_bools:
                    new_param = lines[i+1].split('\n')[0]
                    if new_param == 'True':
                        output = True
                        globals()[str_test] = True
                    elif new_param == 'False':
                        output = False
                        globals()[str_test] = False

                elif str_test in list_chars:
                    new_param = lines[i+1].split('\n')[0]
                    output = new_param
                    globals()[str_test] = new_param
                
                return test, output
    return test, None

def read_until(lines, str_test):
    test, output = find_string(lines, str_test)
    if test == False:
        raise TypeError(f"{str_test} was not found in data file")
    else:
        return output

class parameters:
    def __init__(self,data_file):
            with open(data_file,'r') as f:
                lines = f.readlines()

#######################################################
            mesh_ext = read_until(lines, 'mesh_ext')
            self.mesh_ext = mesh_ext
#######################################################
            path_to_mesh = read_until(lines, 'path_to_mesh')
            self.path_to_mesh = path_to_mesh
#######################################################
            test, nb_bits = find_string(lines, 'nb_bits')
            if not test:
                nb_bits = 64
                type_float = np.float64
            else:
                if nb_bits == 64:
                    type_float = np.float64
                elif nb_bits == 32:
                    type_float = np.float32
                else:
                    raise ValueError(f"nb_bits must be 32 or 64 (not {nb_bits})")
            self.nb_bits = nb_bits
            self.type_float = type_float
#######################################################
            #path_SFEMaNS_env = read_until(lines, 'path_SFEMaNS_env')
            #self.path_SFEMaNS_env = path_SFEMaNS_env
#######################################################
            test, READ_FROM_SUITE = find_string(lines, 'READ_FROM_SUITE')
            if not test:
                READ_FROM_SUITE = False
            self.READ_FROM_SUITE = READ_FROM_SUITE
#######################################################
            field = read_until(lines, 'field')
            self.field = field
#######################################################
            test, D = find_string(lines, 'D')
            if not test:
                D = None
            self.D = D
#######################################################
            test, mesh_type = find_string(lines, 'mesh_type')
            if not test:
                mesh_type = None
            self.mesh_type = mesh_type
#######################################################
            # test, S = find_string(lines, 'S')
            # if not test:
#######################################################
            MF = read_until(lines, 'MF')
            self.MF = MF
#######################################################
            test, opt_mF = find_string(lines, 'opt_mF')
            if not test:
                opt_mF = np.arange(MF)
            self.opt_mF = opt_mF
#######################################################
            should_we_save_Fourier_POD = read_until(lines, 'should_we_save_Fourier_POD')
            self.should_we_save_Fourier_POD = should_we_save_Fourier_POD
#######################################################
            should_we_save_phys_POD = read_until(lines, 'should_we_save_phys_POD')
            self.should_we_save_phys_POD = should_we_save_phys_POD
#######################################################
            test, nb_proc_in_fourier = find_string(lines, 'nb_proc_in_fourier')
            if not test:
                print('Did not specify nb_proc_in_fourier, setting to 1 by default')
                nb_proc_in_fourier = 1
            self.nb_proc_in_fourier = nb_proc_in_fourier
#######################################################
            test, nb_proc_in_axis = find_string(lines, 'nb_proc_in_axis')
            if not test:
                print('Did not specify nb_proc_in_axis, setting to 1 by default')
                nb_proc_in_axis = 1
            self.nb_proc_in_axis = nb_proc_in_axis
####################################################### NOT WORKING
            test, nb_proc_in_meridian = find_string(lines, 'nb_proc_in_meridian')
            if not test:
                print('Did not specify nb_proc_in_meridian, setting to 1 by default')
                nb_proc_in_meridian = 1
            self.nb_proc_in_meridian = nb_proc_in_meridian
#######################################################
            test, is_the_field_to_be_renormalized_by_magnetic_energy = find_string(lines, 'is_the_field_to_be_renormalized_by_magnetic_energy')
            if not test:
                is_the_field_to_be_renormalized_by_magnetic_energy = False
            self.is_the_field_to_be_renormalized_by_magnetic_energy = is_the_field_to_be_renormalized_by_magnetic_energy
#######################################################
            test, is_the_field_to_be_renormalized_by_its_L2_norm = find_string(lines, 'is_the_field_to_be_renormalized_by_its_L2_norm')
            if not test:
                is_the_field_to_be_renormalized_by_its_L2_norm = False
            self.is_the_field_to_be_renormalized_by_its_L2_norm = is_the_field_to_be_renormalized_by_its_L2_norm
#######################################################
            test, should_we_remove_mean_field = find_string(lines, 'should_we_remove_mean_field')
            if not test:
                should_we_remove_mean_field = False
            self.should_we_remove_mean_field = should_we_remove_mean_field
#######################################################
#            test, should_mean_field_computation_include_mesh_sym = find_string(lines, 'should_mean_field_computation_include_mesh_sym')
#            if not test:
#                should_mean_field_computation_include_mesh_sym = True
#            self.should_mean_field_computation_include_mesh_sym = should_mean_field_computation_include_mesh_sym
#######################################################
            test, should_mean_field_be_axisymmetric = find_string(lines, 'should_mean_field_be_axisymmetric')
            if not test:
                should_mean_field_be_axisymmetric = False
            self.should_mean_field_be_axisymmetric = should_mean_field_be_axisymmetric
#######################################################
            test, should_we_modify_weights = find_string(lines, 'should_we_modify_weights')
            if not test:
                should_we_modify_weights = False
            self.should_we_modify_weights = should_we_modify_weights
#######################################################
            test, should_we_penalize_divergence = find_string(lines, 'should_we_penalize_divergence')
            if test and self.D != 3:
                raise ValueError(f"Cannot penalize divergence if entry dimension is {self.D}")
            if not test:
                should_we_penalize_divergence = False
            self.should_we_penalize_divergence = should_we_penalize_divergence
#######################################################
            if self.should_we_penalize_divergence:
                test, penal_div = find_string(lines, 'penal_div')
                if not test:
                    penal_div = 1.0
                self.penal_div = penal_div
#######################################################
            test, directory_scalar_for_weights = find_string(lines, 'directory_scalar_for_weights')
            if not test:
                directory_scalar_for_weights = ''
            self.directory_scalar_for_weights = directory_scalar_for_weights
#######################################################
            test, should_we_add_mesh_symmetry = find_string(lines, 'should_we_add_mesh_symmetry')
            if not test:
                should_we_add_mesh_symmetry = False
            self.should_we_add_mesh_symmetry = should_we_add_mesh_symmetry
#######################################################
            test, should_we_restrain_to_symmetric = find_string(lines, 'should_we_restrain_to_symmetric')
            if not test:
                should_we_restrain_to_symmetric = False
            self.should_we_restrain_to_symmetric = should_we_restrain_to_symmetric
#######################################################
            test, should_we_restrain_to_antisymmetric = find_string(lines, 'should_we_restrain_to_antisymmetric')
            if not test:
                should_we_restrain_to_antisymmetric = False
            self.should_we_restrain_to_antisymmetric = should_we_restrain_to_antisymmetric
#######################################################
            test, type_sym = find_string(lines, 'type_sym')
            if not test:
                type_sym = None
            self.type_sym = type_sym
#######################################################
            test, should_we_combine_with_shifted_data = find_string(lines, 'should_we_combine_with_shifted_data')
            if not test:
                should_we_combine_with_shifted_data = False
            self.should_we_combine_with_shifted_data = should_we_combine_with_shifted_data
#######################################################
            test, shift_angle = find_string(lines, 'shift_angle')
            if not test:
                shift_angle = []
            self.shift_angle = shift_angle
#######################################################
            test, number_shifts = find_string(lines, 'number_shifts')
            if not test:
                number_shifts = 1
            self.number_shifts = number_shifts
#######################################################
            test, should_we_save_all_fourier_pod_modes = find_string(lines, 'should_we_save_all_fourier_pod_modes')
            if not test:
                should_we_save_all_fourier_pod_modes = False
            self.should_we_save_all_fourier_pod_modes = should_we_save_all_fourier_pod_modes
#######################################################
            fourier_pod_modes_to_save = read_until(lines, 'fourier_pod_modes_to_save')
            self.fourier_pod_modes_to_save = fourier_pod_modes_to_save
#######################################################
            test, should_we_save_all_phys_pod_modes = find_string(lines, 'should_we_save_all_phys_pod_modes')
            if not test:
                should_we_save_all_phys_pod_modes = False
            self.should_we_save_all_phys_pod_modes = should_we_save_all_phys_pod_modes
#######################################################
            phys_pod_modes_to_save = read_until(lines, 'phys_pod_modes_to_save')
            self.phys_pod_modes_to_save = phys_pod_modes_to_save
#######################################################
            path_to_suites = read_until(lines, 'path_to_suites')
            self.path_to_suites = path_to_suites
#######################################################
            paths_to_data = read_until(lines, 'paths_to_data')
            self.paths_to_data = paths_to_data
#######################################################
            output_path = read_until(lines, 'output_path')
            self.output_path = output_path
#######################################################
            test, output_file_name = find_string(lines, 'output_file_name')
            if not test:
                output_file_name = ''
            self.output_file_name = output_file_name
#######################################################
            test, name_job_output = find_string(lines, 'name_job_output')
            if not test:
                name_job_output = 'job_out.txt'
            self.name_job_output = name_job_output
#######################################################
            test, should_we_extract_latents = find_string(lines, 'should_we_extract_latents')
            if not test:
                should_we_extract_latents = True
            self.should_we_extract_latents = should_we_extract_latents
#######################################################
            test, should_we_save_phys_correlation = find_string(lines, 'should_we_save_phys_correlation')
            if not test:
                should_we_save_phys_correlation = False
            self.should_we_save_phys_correlation = should_we_save_phys_correlation
#######################################################
            test, should_we_extract_modes = find_string(lines, 'should_we_extract_modes')
            if not test:
                should_we_extract_modes = True
            self.should_we_extract_modes = should_we_extract_modes
#######################################################
            test, save_bins_format = find_string(lines, 'save_bins_format')
            if not test:
                save_bins_format = True
            self.save_bins_format = save_bins_format
#######################################################
            test, bins_format = find_string(lines, 'bins_format')
            if not test:
                bins_format = 'fourier'
            self.bins_format = bins_format
#######################################################
            test, do_post_tests = find_string(lines, 'should_we_do_all_post_tests')
            if not test:
                do_post_tests = False
            self.do_post_tests = do_post_tests
#######################################################
            test, read_from_gauss = find_string(lines, 'read_from_gauss')
            if not test:
                read_from_gauss = False
            self.read_from_gauss = read_from_gauss




#######################################################
#######################################################
#######################################################
#######################################################
#######################################################


# def global_parameters(data_file,list_ints,list_several_ints,list_floats,list_several_floats,list_bools,
# list_chars,list_several_chars,list_several_list_chars,list_fcts):
#     with open(data_file,'r') as f:
#         lines = f.readlines()

#     mesh_ext = read_until()

#     for i,line in enumerate(raw_lines):
#         line = line.split('\n')[0]
#         if '===' in line:
#             new_name = line.split('===')[1]
#             new_param = raw_lines[i+1].split('\n')[0]
#             if new_name in list_several_chars:
#                 list_new_params = []
#                 j = i+1
            #     new_param = raw_lines[j].split('\n')[0]
            #     while len(new_param) > 0:
            #         list_new_params.append(new_param)
            #         j += 1
            #         new_param = raw_lines[j].split('\n')[0]
            #     globals()[new_name] = list_new_params
            #     i = j
            # elif new_name in list_several_list_chars:
            #     list_new_params = []
            #     j = i+1
            #     new_param = raw_lines[j].split('\n')[0]
            #     while len(new_param) > 0:
            #         list_new_params.append(new_param.split(','))
            #         j += 1
            #         new_param = raw_lines[j].split('\n')[0]
            #     globals()[new_name] = list_new_params
                # i = j
            # elif new_name in list_several_ints:
            #     new_param = raw_lines[i+1].split('\n')[0]
            #     new_param = new_param.split(',')
            #     list_new_params = np.empty(len(new_param),dtype=int)
            #     for k,num in enumerate(new_param):
            #         list_new_params[k] = int(num)
            #     globals()[new_name] = list_new_params
            #     list_new_params = np.array(list_new_params,dtype=int)
            # elif new_name in list_several_floats:
            #     new_param = raw_lines[i+1].split('\n')[0]
            #     new_param = new_param.split(',')
            #     list_new_params = np.empty(len(new_param),dtype=float)
            #     for k,num in enumerate(new_param):
            #         list_new_params[k] = float(num)
            #     globals()[new_name] = list_new_params
            #     list_new_params = np.array(list_new_params,dtype=float)
            # elif new_name in list_fcts:
            #     if new_name == "fct_for_custom_field":
            #         new_param = raw_lines[i+1].split('\n')[0]
            #         new_param = new_param.split(',')[0]

            #         fct_for_custom_field = globals()[new_param]
                    # globals()[new_name] = globals()[new_param]
    #         else:
    #             new_param = raw_lines[i+1].split('\n')[0]
    #             if new_name in list_ints:
    #                 globals()[new_name] = int(new_param)
    #             elif new_name in list_floats:
    #                 globals()[new_name] = float(new_param)
    #             elif new_name in list_bools:
    #                 if new_param == 'True':
    #                     globals()[new_name] = True
    #                 elif new_param == 'False':
    #                     globals()[new_name] = False
    #             elif new_name in list_chars:
    #                 globals()[new_name] = new_param

    # elms_ints = [globals()[quantity_int] for quantity_int in list_ints]
    # elms_lists_ints = [globals()[several_ints] for several_ints in list_several_ints]
    # elms_floats = [globals()[quantity_float] for quantity_float in list_floats]
    # elms_lists_floats = [globals()[several_floats] for several_floats in list_several_floats]
    # elms_bools = [globals()[quantity_bool] for quantity_bool in list_bools]
    # elms_chars = [globals()[quantity_char] for quantity_char in list_chars]
    # elms_lists_chars = [globals()[several_chars] for several_chars in list_several_chars]
    # elms_lists_lists_chars = [globals()[several_lists_chars] for several_lists_chars in list_several_list_chars]
    # # elms_fcts = [globals()[quantity_func] for quantity_func in list_fcts]
    # elms_fcts = []

    # all_parameters = parameters(list_ints,elms_ints,list_several_ints,elms_lists_ints,list_floats,elms_floats,list_several_floats,elms_lists_floats,
    #                             list_bools,elms_bools,
    #                             list_chars,elms_chars,list_several_chars,elms_lists_chars,list_several_list_chars,elms_lists_lists_chars,
    #                             list_fcts,elms_fcts)

    # return all_parameters
