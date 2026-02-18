from brian2 import StateMonitor

bcpnn_layer_variables = ['V', 'S_j', 'Z_j', 'E_j', 'P_j', 'beta', 'g_ex', 'g_inh', 'I_beta', 'I_on']
bcpnn_synapse_variables = ['w', 'Z_i', 'E_i', 'P_i', 'E_syn', 'P_syn']

def get_inh_synapses(n_hyper, n_mini):

    source = []
    target = []

    for i_hyper in range(n_hyper):
        hyper_offset = i_hyper * n_hyper
        for source_mini in range(hyper_offset, hyper_offset+n_mini):
            for target_mini in range(hyper_offset, hyper_offset+n_mini):
                if source_mini != target_mini:
                    source.append(source_mini)
                    target.append(target_mini)
    
    return source, target

def get_BCPNN_statemon(neurons, record):
    return StateMonitor(neurons, bcpnn_layer_variables, record=record)

def get_BCPNN_synmon(synapse, record):
    return StateMonitor(synapse, bcpnn_synapse_variables, record=record)