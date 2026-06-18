from brian2 import *

tully_namespace = {
    # NEURON MODEL
    'C_m': 250 * pF, # membrane capacitance
    'g_L': 16.67 * nS, # leak conductance
    'E_L': -70 * mV, # leak reversal potential
    'E_ex': 0 * mV, # excitatory reversal potential
    'E_inh': -75 * mV, # inhibiory reversal potential
    'phi': 0 * pA, #50 * pA # current scaling factor
    'V_th': -55 * mV, # membrane voltage threshold
    'V_res': -60 * mV, # membrane reset potential
    't_ref': 2 * ms, # refractory period
    'sim_dt': 0.1 * ms, # time resolution

    'min_num': 10e-10, # minimum float value
    'dI': -0.3 * nA, # external current

    # CHANNEL MODEL
    'tau_z': 10 * ms, # Z trace time constant
    'tau_e': 100 * ms, # E trace time constant
    'tau_p': 3000 * ms, # P trace time constant
    'f_max': 20 * Hz, # highest firing rate
    'f_min': 1 * Hz, # min firing rate
    'epsilon': 0.0001, # min bcpnn probability
    't_spike': 0.1 * ms, # spike duration

    # SYNAPSE MODEL
    'g_max': 2 * nS, # peak conductance
    'tau_ex': 0.2 * ms, # alpha rise time for excitatory input
    'tau_inh': 2 * ms, # alpha rise time for inhibitory neurons
    'd': 0.1 * ms, # transmission delay
    'K': 1, # learning rate -> 0 to freeze plasticity
    
    # INPUT
    'n_ex': 30, # number of independent poisson processes per neuron
    'w_ex': 10.75 * nS, # weight per process
    'r_ex': 30 * Hz, # Poisson input firing rate
    'tau_input': 0.7 * ms # input EPSP time constant

}

tully_equations = {
    # NEURON EQUATIONS
    'eqs_rec': '''
    # POSTSYNAPTIC (j) TRACES
    dS/dt = -S/sim_dt : 1
    dZ_j/dt = (S/(f_max*t_spike) - Z_j + epsilon)/tau_z : 1
    dE_j/dt = (Z_j - E_j)/tau_e : 1
    dP_j/dt = (K*(E_j-P_j))/tau_p : 1

    # bias
    beta = log(P_j) : 1
    I_beta = phi*beta : amp

    # on-switch
    b_on = stim_ta(t,i) : 1
    dg_stim/dt = -g_stim/tau_input : siemens
    dg_alpha/dt = (g_stim-g_alpha)/tau_input : siemens
    I_stim = b_on * g_stim * (V_m-E_ex) : amp

    # total voltage
    g_ex : siemens # summed over all excitatory synapses
    g_inh : siemens # summed over all inhibitory synapses
    dV_m/dt = (g_L*(V_m-E_L)+g_ex*(V_m-E_ex)+g_inh*(V_m-E_inh)+I_beta + I_stim)/-C_m : volt (unless refractory)
    ''',
    'reset_rec': '''
    V_m = V_res
    S = 1
    ''',
    'threshold_rec': 'V_m >= V_th',
    'refractory_rec': 't_ref',
    
    # SYNAPSE EQUATIONS
    'bcpnn_syn_model': '''
    # CONDUCTANCES ----------------------------------------------
    w_g = abs(w) * g_max : siemens # weighted maximum condunctance
    # EXCITATORY ------------------------------------------------
    dS_ex/dt = -S_ex/tau_ex : 1 (clock-driven) # excitatory conducting window
    dalpha_ex/dt = (S_ex-alpha_ex)/tau_ex : 1  (clock-driven)
    g_ex_post = b_ex * w_g * alpha_ex : siemens (summed)
    # INHIBITORY ------------------------------------------------
    dS_inh/dt = -S_inh/tau_inh : 1 (clock-driven) # inhibitory conducting window
    dalpha_inh/dt = (S_inh-alpha_inh)/tau_inh : 1 (clock-driven)
    g_inh_post = (1-b_ex) * w_g * alpha_inh : siemens (summed)

    # PRESYNAPTIC (i) TRACES ------------------------------------
    dS_i/dt = -S_i/sim_dt : 1 (clock-driven)
    dZ_i/dt = (S_i/(f_max*t_spike) - Z_i + epsilon)/tau_z : 1 (clock-driven)
    dE_i/dt = (Z_i - E_i)/tau_e : 1 (clock-driven)
    dP_i/dt = (K*(E_i-P_i))/tau_p : 1 (clock-driven)

    # BCPNN SYNAPSE ---------------------------------------------
    dE_syn/dt = (Z_i*Z_j_post - E_syn)/tau_e : 1 (clock-driven)
    dP_syn/dt = (K*(E_syn-P_syn))/tau_p : 1 (clock-driven)
    w = log(P_syn/(P_i*P_j_post)) : 1 (constant over dt)
    b_ex = int(w > 0) : 1 # 1 if synapse is excitatory, 0 if inhibitory
    ''',

    'bcpnn_syn_on_pre': '''
    S_i = 1
    S_ex = 1
    S_inh = 1
    '''
}