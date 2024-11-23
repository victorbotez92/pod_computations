def write_job_output(path,message):
    with open(path, 'r') as f:
        content = f.read()
    with open(path, 'w') as f:
        f.write(content+'\n'+message)

def indiv_ranks(par):
    rank_axis = par.rank//(par.nb_proc_in_fourier*par.nb_proc_in_meridian*par.nb_proc_in_dimension)
    rank_minus_axis = par.rank%(par.nb_proc_in_fourier*par.nb_proc_in_meridian*par.nb_proc_in_dimension)
    rank_meridian = rank_minus_axis//(par.nb_proc_in_fourier*par.nb_proc_in_dimension)
    rank_minus_axis_minus_meridian = rank_minus_axis%(par.nb_proc_in_fourier*par.nb_proc_in_dimension)
    rank_dimension,rank_fourier = rank_minus_axis_minus_meridian//par.nb_proc_in_fourier,rank_minus_axis_minus_meridian%par.nb_proc_in_fourier
    return rank_axis,rank_fourier,rank_meridian,rank_dimension

def invert_rank(rank_fourier,rank_axis,rank_meridian,rank_dimension,par):
    rank_minus_axis_minus_meridian = par.nb_proc_in_fourier*rank_dimension + rank_fourier
    rank_minus_axis = rank_meridian*(par.nb_proc_in_fourier*par.nb_proc_in_dimension) + rank_minus_axis_minus_meridian
    rank = rank_minus_axis + rank_axis*(par.nb_proc_in_fourier*par.nb_proc_in_meridian*par.nb_proc_in_dimension)
    return rank