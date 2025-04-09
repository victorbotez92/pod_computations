import numpy as np
from template_fields_to_remove import *

class parameters:
    def __init__(self,list_ints,elms_ints,list_several_ints,elms_lists_ints,list_floats,elms_floats,list_several_floats,elms_lists_floats,list_bools,
                                elms_bools,list_chars,elms_chars,list_several_chars,elms_lists_chars,list_several_list_chars,elms_lists_lists_chars,
                                list_fcts,elms_fcts):
        for i in range(len(list_ints)):
            setattr(self,list_ints[i],elms_ints[i])
        for i in range(len(list_several_ints)):
            setattr(self,list_several_ints[i],elms_lists_ints[i])
        for i in range(len(list_floats)):
            setattr(self,list_floats[i],elms_floats[i])
        for i in range(len(list_several_floats)):
            setattr(self,list_several_floats[i],elms_lists_floats[i])
        for i in range(len(list_bools)):
            setattr(self,list_bools[i],elms_bools[i])
        for i in range(len(list_chars)):
            setattr(self,list_chars[i],elms_chars[i])
        for i in range(len(list_several_chars)):
            setattr(self,list_several_chars[i],elms_lists_chars[i])
        for i in range(len(list_several_list_chars)):
            setattr(self,list_several_list_chars[i],elms_lists_lists_chars[i])
        for i in range(len(list_fcts)):
            setattr(self,list_fcts[i],elms_fcts[i])


def global_parameters(data_file,list_ints,list_several_ints,list_floats,list_several_floats,list_bools,
list_chars,list_several_chars,list_several_list_chars,list_fcts):
    with open(data_file,'r') as f:
        raw_lines = f.readlines()

    for i,line in enumerate(raw_lines):
        line = line.split('\n')[0]
        if '===' in line:
            new_name = line.split('===')[1]
            new_param = raw_lines[i+1].split('\n')[0]
            if new_name in list_several_chars:
                list_new_params = []
                j = i+1
                new_param = raw_lines[j].split('\n')[0]
                while len(new_param) > 0:
                    list_new_params.append(new_param)
                    j += 1
                    new_param = raw_lines[j].split('\n')[0]
                globals()[new_name] = list_new_params
                i = j
            elif new_name in list_several_list_chars:
                list_new_params = []
                j = i+1
                new_param = raw_lines[j].split('\n')[0]
                while len(new_param) > 0:
                    list_new_params.append(new_param.split(','))
                    j += 1
                    new_param = raw_lines[j].split('\n')[0]
                globals()[new_name] = list_new_params
                i = j
            elif new_name in list_several_ints:
                new_param = raw_lines[i+1].split('\n')[0]
                new_param = new_param.split(',')
                list_new_params = np.empty(len(new_param),dtype=int)
                for k,num in enumerate(new_param):
                    list_new_params[k] = int(num)
                globals()[new_name] = list_new_params
                list_new_params = np.array(list_new_params,dtype=int)
            elif new_name in list_several_floats:
                new_param = raw_lines[i+1].split('\n')[0]
                new_param = new_param.split(',')
                list_new_params = np.empty(len(new_param),dtype=float)
                for k,num in enumerate(new_param):
                    list_new_params[k] = float(num)
                globals()[new_name] = list_new_params
                list_new_params = np.array(list_new_params,dtype=float)
            elif new_name in list_fcts:
                if new_name == "fct_for_custom_field":
                    new_param = raw_lines[i+1].split('\n')[0]
                    new_param = new_param.split(',')[0]
                    print(new_param)

                    fct_for_custom_field = globals()[new_param]
                    # globals()[new_name] = globals()[new_param]
            else:
                new_param = raw_lines[i+1].split('\n')[0]
                if new_name in list_ints:
                    globals()[new_name] = int(new_param)
                elif new_name in list_floats:
                    globals()[new_name] = float(new_param)
                elif new_name in list_bools:
                    if new_param == 'True':
                        globals()[new_name] = True
                    elif new_param == 'False':
                        globals()[new_name] = False
                elif new_name in list_chars:
                    globals()[new_name] = new_param

    elms_ints = [globals()[quantity_int] for quantity_int in list_ints]
    elms_lists_ints = [globals()[several_ints] for several_ints in list_several_ints]
    elms_floats = [globals()[quantity_float] for quantity_float in list_floats]
    elms_lists_floats = [globals()[several_floats] for several_floats in list_several_floats]
    elms_bools = [globals()[quantity_bool] for quantity_bool in list_bools]
    elms_chars = [globals()[quantity_char] for quantity_char in list_chars]
    elms_lists_chars = [globals()[several_chars] for several_chars in list_several_chars]
    elms_lists_lists_chars = [globals()[several_lists_chars] for several_lists_chars in list_several_list_chars]
    # elms_fcts = [globals()[quantity_func] for quantity_func in list_fcts]
    elms_fcts = [fct_for_custom_field]

    all_parameters = parameters(list_ints,elms_ints,list_several_ints,elms_lists_ints,list_floats,elms_floats,list_several_floats,elms_lists_floats,
                                list_bools,elms_bools,
                                list_chars,elms_chars,list_several_chars,elms_lists_chars,list_several_list_chars,elms_lists_lists_chars,
                                list_fcts,elms_fcts)

    return all_parameters