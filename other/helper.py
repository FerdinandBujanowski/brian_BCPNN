def get_inh_synapses(n_hyper, n_mini):

    source = []
    target = []

    for i_hyper in range(n_hyper):
        hyper_offset = i_hyper * n_hyper
        for source_mini in range(hyper_offset, hyper_offset+8):
            for target_mini in range(hyper_offset, hyper_offset+8):
                if source_mini != target_mini:
                    source.append(source_mini)
                    target.append(target_mini)
    
    return source, target