def write_job_output(path,message):
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