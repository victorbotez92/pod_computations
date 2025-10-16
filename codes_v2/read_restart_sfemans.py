from os import listdir
from os.path import isfile, join
import numpy as np
import sys

sys.path.append("/ccc/cont003/home/limsi/bousquer/einops")
from einops import rearrange, einsum

def skip_record(file,n):
    """
    file : file pointer (with open(path,'rb') as file)
    n : number of record to skip
    """
    for _ in range(n):
         # skip first line : read record lenght, go to the next record
        record_length_bytes = file.read(4)
        record_length = np.frombuffer(record_length_bytes, dtype=np.int32)[0]
        file.seek(record_length+4,1)
    return 

   

def read_in_suite(path,mF, first_offset, n_components=6, record_stack_lenght=7):
    """
    path (str)                : path to the specific suite file
    mF (int)                  : specific fourier mode to read in the underlying file
    first_offset (int)        : position of the field, in the record stack, see below 
    n_components (int)        : number of components in fourier space, == 2*d with d the field dimension
    record_stack_lenght (int) : number of record (lines) written for each fourier mode
    
    #eg of suite_ns file structure (without LES and Multifluid options):
    
                    X : first line
             # Field stack begin
                    n : mode number = 0
                    u : field u(t)
                    u(m-1) : field u(t-dt)
                    p : field p(t)
                    p(m-1) : field p(t-dt)
                    icr_p : correction of field p(t)
                    incr_p(m-1) : correction of field p(t-dt)
            # Field stack end
                    n : mode number = 1
                    ...
                    
        eg : for u, first_offset=2, record_stack_lenght=7
        eg : for p, first_offset=4, record_stack_lenght=7
        eg : for les record_stack_lenght=8
        
    #eg of suite_maxwell file structure(without DNS and Multifluid) :
    
                    X : first line
             # Field stack begin
                    n : mode number = 0
                    H : field H(t)
                    H(m-1) : field H(t-dt)
                    B : field B(t)
                    B(m-1) : field B(t-dt)
                    phi : field phi(t)
                    phi(m-1) : field phi(t-dt)
            # Field stack end
                    n : mode number = 1
                    ...
                    
        eg : for H, first_offset=2, record_stack_lenght=7
        eg : for B, first_offset=4, record_stack_lenght=7
        
    returns :
    the raw field read in the suite file, on nodes points
    
    """
    with open(path,'rb') as file:
        # go to the given mF record
        skip_record(file,first_offset+mF*record_stack_lenght)
        
        # get record lenght
        record_length_bytes = file.read(4)
        record_length = np.frombuffer(record_length_bytes, dtype=np.int32)[0]
        
        #assuming double precision
        num_elements = record_length // 8
        n_nodes = num_elements//n_components
        field = np.fromfile(file,dtype=np.float64,count=num_elements).reshape(n_nodes,n_components,order='F')
        
    return field

def get_data_from_suites(path_to_all_suites,path_to_mesh,mF_to_read,S,field_name_in_file="u",record_stack_lenght=7, get_gauss_points=True,stack_domains=True):
    """
    path_to_all_suites (str) : path to the directory where all the suites are stored
    path_to_mesh (str)       : path to the directory where Xmesh_jj are stored
    mF_to_read (int)         : specific fourier mode to read    
    S (int)                  : number of domains
    field_name_in_file (str) : field to read, must be in ["u","p","H","B"]
    record_stack_lenght(int) : see 'read_in_suite' function
    get_gauss_points (bool)  : if true the field is evaluated on gauss points
    stack_domains (bool)     : if true the domains are stacked along a single array direction
     
    returns field :
    the underlying field, for mode mF_to_read.
    """
    # select between suite_ns and suite_maxwell
    if field_name_in_file == "u":
        suite_kind="ns"
        mesh_kind = "vv"
        first_offset=2
        nb_components=6
        
    elif field_name_in_file == "p" :
        suite_kind="ns"
        mesh_kind = "vv"
        first_offset=4
        nb_components=2
        
    elif field_name_in_file == "H" :
        suite_kind="maxwell"
        mesh_kind = "H"
        first_offset=2
        nb_components=6
        
    elif field_name_in_file == "B": 
        suite_kind="maxwell"
        mesh_kind = "H"
        first_offset=4
        nb_components=6
        
    else:
        print("Field",field_name_in_file,"not found, or not implemented")
        return
    
    #get all suite file str for the given suite kind 
    suite_files = np.sort([f for f in listdir(path_to_all_suites) if isfile(join(path_to_all_suites, f)) if suite_kind in f])
    Nfile = suite_files.size
    Nt = Nfile//S
    fields=[]
    for i in range(Nfile):
        path = path_to_all_suites+"/"+suite_files[i]
        f = read_in_suite(path, mF_to_read, first_offset, nb_components, record_stack_lenght)
        fields.append(f)  
    TEMP=[]
    for s in range(S):
        TEMP.append( np.asarray(fields[s*Nt:(s+1)*Nt])  )
    
    if get_gauss_points :
        #get mesh info
        ME = [ read_mesh_info(path_to_mesh+f"/{mesh_kind}mesh_info_S{s:04d}.txt")[2] for s in range(S) ]  
        n_w = read_mesh_info(path_to_mesh+f"/{mesh_kind}mesh_info_S0000.txt")[0]
        l_G = read_mesh_info(path_to_mesh+f"/{mesh_kind}mesh_info_S0000.txt")[1]
        
        
        # get global connectivity, and test function weights
        mesh_ext= [f for f in listdir(path_to_mesh) if isfile(join(path_to_mesh, f)) if '.FEM' in f][0].split(".")[-2]+".FEM"
        mesh_jj = [ np.fromfile(path_to_mesh+f"/{mesh_kind}mesh_jj_S{s:04d}."+mesh_ext,dtype=np.int32).reshape(n_w,ME[s],order="F") 
                   for s in range(S) ] 
        mesh_ww = [ np.fromfile(path_to_mesh+f"/{mesh_kind}mesh_gauss_ww_S{s:04d}."+mesh_ext,dtype=np.float64).reshape(n_w,l_G,order="F") 
                   for s in range(S) ] 
        
        # node point to gauss points
        TEMP_gauss =[]            
        for s in range(S):
            #arange field by triangle
            X = np.asarray( [TEMP[s][:,mesh_jj[s][i]-1,:] for i in range(n_w)] )
            field_gauss = einsum(X,mesh_ww[s],'nw t me d, nw l_G -> t me l_G d ')
            TEMP_gauss.append( rearrange(field_gauss, "t me l_G d -> t (me l_G) d") )
        TEMP = TEMP_gauss
    
    if stack_domains:
        output=[]
        for t in range(Nt):
            output.append(np.vstack([TEMP[s][t] for s in range(S)]))
        # data has now shape (T,N_node_points_tot), domains are stacked : 0 then 1 then ... 
        return  np.asarray(output)
    else: 
        return TEMP 
    
def read_mesh_info(path):
    with open(path) as file:
        line = file.readlines()
        values = line[1::2][:4]
        n_w = int(values[0])
        l_G = int(values[1])
        me  = int(values[2])
        n_p = int(values[3])
    return n_w,l_G,me,n_p

if __name__ == "__main__":

    path_to_suite = f"./DNS_MHD_TM73/"
    path_to_mesh="./TM73_mesh_bin/Hmesh/"
    mF_to_read = 12
    S = 3
    field = get_data_from_suites(path_to_suite,
                                 path_to_mesh,
                                 mF_to_read,
                                 S,
                                 field_name_in_file="B",
                                 record_stack_lenght=7,
                                 get_gauss_points=True,
                                 stack_domains=True)
    # output field has shape (time,n_mesh_point,d=6) for 3D field if stack domains is true, else
    # field is a list containing (time,n_mesh_point_per_domain,d=6)
    # d=2 for scalar field
    # d order is like in sfemans : ur_c, ur_s, uth_c, uth_s, uz_c, uz_s