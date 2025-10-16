def write_job_output(par,message,list_ranks = [0]):
    path = par.path_to_job_output
    if par.rank in list_ranks:
        with open(path, 'r') as f:
            content = f.read()
        with open(path, 'w') as f:
            f.write(content+'\n'+message)

def indiv_ranks(par):
    rank_axis = par.rank//(par.nb_proc_in_fourier*par.nb_proc_in_meridian)
    rank_minus_axis = par.rank%(par.nb_proc_in_fourier*par.nb_proc_in_meridian)
    rank_meridian,rank_fourier = rank_minus_axis//par.nb_proc_in_fourier,rank_minus_axis%par.nb_proc_in_fourier
    return rank_axis,rank_fourier,rank_meridian

def invert_rank(rank_fourier,rank_axis,rank_meridian,par):
    rank_minus_axis = par.nb_proc_in_fourier*rank_meridian + rank_fourier
    rank = rank_minus_axis + rank_axis*(par.nb_proc_in_fourier*par.nb_proc_in_meridian*par.nb_proc_in_axis)//2
    return rank


import psutil

def print_memory_usage(par, tag=""):
    print('inside memory usage')
    process = psutil.Process()
    mem_info = process.memory_info()
    total = psutil.virtual_memory().total / (1024 ** 3)
    used = psutil.virtual_memory().used / (1024 ** 3)
    available = psutil.virtual_memory().available / (1024 ** 3)
    
    # print(f"\n=== Memory report {tag} ===")
    # print(f"Process memory used: {mem_info.rss / (1024 ** 3):.2f} GB")
    # print(f"System memory used:  {used:.2f} GB / {total:.2f} GB total")
    # print(f"System memory free:  {available:.2f} GB\n")

    write_job_output(par, f"\n=== Memory report {tag} ===")
    write_job_output(par, f"Process memory used: {mem_info.rss / (1024 ** 3):.2f} GB")
    write_job_output(par, f"System memory used:  {used:.2f} GB / {total:.2f} GB total")
    write_job_output(par, f"System memory free:  {available:.2f} GB\n")
