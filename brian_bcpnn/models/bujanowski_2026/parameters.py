from brian2 import * 

# SIMULATION
SIM_PARAMS = '''
sim_dt : second
min_num : 1
'''
SIM_PARAM_INIT = {
    'sim_dt': 1 * ms, # time resolution
    'min_num': 10e-10 # minimum float value
}

# NEURON MODEL
NEURON_PARAMS = '''
C_m : farad
g_L : siemens
E_L : volt
E_ex : volt
E_inh : volt
phi : amp
V_th : volt
V_res : volt
t_ref : second
dI : amp
'''
NEURON_PARAM_INIT = {
    'C_m': 250 * pF, # membrane capacitance
    'g_L' : 16.67 * nS, # leak conductance
    'E_L': -70 * mV, # leak reversal potential
    'E_ex': 0 * mV, # excitatory reversal potential
    'E_inh': -75 * mV, # inhibiory reversal potential
    'phi': 0 * pA, #50 * pA # current scaling factor
    'V_th': -55 * mV, # membrane voltage threshold
    'V_res': -60 * mV, # membrane reset potential
    't_ref': 2 * ms, # refractory period
    'dI': -1 * nA # external current
}

# CHANNEL MODEL
TRACE_PARAMS = '''
tau_z : second
tau_e : second
tau_p : second
f_max : Hz
epsilon : Hz
t_spike : second
K : 1
'''
TRACE_PARAM_INIT = {
    'tau_z': 10 * ms, # Z trace time constant
    'tau_e': 100 * ms, # E trace time constant
    'tau_p': 10000 * ms, # P trace time constant
    'f_max': 20 * Hz, # highest firing rate
    'epsilon': 0 * Hz, # lowest firing rate
    't_spike': 1 * ms, # spike duration
    'K': 1 # learning rate -> 0 to freeze plasticity
}
TRACE_PARAM_INIT['epsilon'] = 1/(TRACE_PARAM_INIT['f_max']*TRACE_PARAM_INIT['tau_p']) * Hz


# SYNAPSE MODEL
SYNAPSE_PARAMS = '''
g_max : siemens
tau_ex : second
tau_inh : second
d : second
'''
SYNAPSE_PARAM_INIT = {
    'g_max': 2 * nS, # peak conductance
    'tau_ex': 2 * ms, # alpha rise time for excitatory input
    'tau_inh': 20 * ms, # alpha rise time for inhibitory neurons
    'd': 1 * ms, # transmission delay
}
