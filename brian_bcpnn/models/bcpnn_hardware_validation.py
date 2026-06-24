from brian2 import *

# MODEL PARAMETERS ---------------------------------------------------
model_namespace = {
    # NEURON MODEL ---------------------------------------------------
    'C_m': 250 * pF, # membrane capacitance
    'g_L': 16.67 * nS, # leak conductance
    'E_L': -70 * mV, # leak reversal potential
    'E_ex': 0 * mV, # excitatory reversal potential
    'E_inh': -75 * mV, # inhibiory reversal potential
    'phi': 50 * pA, # current scaling factor (first = 0*pA in paper)
    'V_th': -55 * mV, # membrane voltage threshold
    'V_res': -60 * mV, # membrane reset potential
    't_ref': 2 * ms, # refractory period
    'sim_dt': 0.1 * ms, # time resolution

    # CHANNEL MODEL --------------------------------------------------
    'tau_z': 10 * ms, # Z trace time constant
    'tau_e': 100 * ms, # E trace time constant
    'tau_p': 1000 * ms, # P trace time constant
    'f_max': 20 * Hz, # highest firing rate
    'f_min': 1 * Hz, # min firing rate
    'epsilon': 0.05, # min bcpnn probability (f_min/f_max)
    't_spike': 0.1 * ms, # spike duration

    # SYNAPSE MODEL --------------------------------------------------
    'g_max': 2 * nS, # peak conductance
    'tau_ex': 0.2 * ms, # alpha rise time for excitatory input
    'tau_inh': 2 * ms, # alpha rise time for inhibitory neurons
    'd': 0.1 * ms, # transmission delay
    'kappa': 1, # learning rate -> 0 to freeze plasticity
    
    # INPUT ----------------------------------------------------------
    'n_ex': 30, # number of independent poisson processes per neuron
    'w_ex': 10.75 * nS, # weight per process
    'r_ex': 30 * Hz, # Poisson input firing rate
    'tau_input': 0.7 * ms # input EPSP time constant
}

# NEURON MODEL -------------------------------------------------------
simplistic_neuron = '''
# EXTERNAL STIMULATION -----------------------------------------------
    b_on = stim_ta(t,i) : 1
'''
neuron_equations = '''

# POST-SYNAPTIC TRACES -----------------------------------------------
    dS/dt = -S/sim_dt : 1
    dZ/dt = (S/(f_max*t_spike) - Z + epsilon) / tau_z : 1
    dE/dt = (Z-E)/tau_e : 1
    dP/dt = (K*(E-P))/tau_p : 1

# BIAS ---------------------------------------------------------------
    beta = log(P) : 1
    I_beta = phi*beta : amp

# EXTERNAL STIMULATION -----------------------------------------------
    b_on = stim_ta(t,i) : 1
    dg_stim/dt = -g_stim/tau_input : siemens
    I_stim = b_on * g_stim * (V_m-E_ex) : amp

# MEMBRANE VOLTAGE ---------------------------------------------------
    g_ex : siemens # summed over all ex synapses (1 in this case)
    g_inh : siemens # summed over all inh synapses (0 in this case)
    dV/dt = (
          g_L   * (V_m-E_L)   # leak current
        + g_ex  * (V_m-E_ex)  # excitatory current
        + g_inh * (V_m-E_inh) # inhibitory current
        + I_beta              # intrinsic excitability
        + I_stim              # external stimulation
    )/-C_m : volt (unless refractory)
'''

# set up brian2 NeuronGroup object:
# in this case, each neuron only consists of a boolean,
# which at each time step monitors whether the neuron is currently spiking or not
# (NB: this boolean is NOT equal to a singular spike, but rather a flag - currently
# spiking or not spiking)
neurons = NeuronGroup(N=2, model=simplistic_neuron)

# set up timed array

# SYNAPSE MODEL ------------------------------------------------------
synapse_model = '''
'''

# set up synapse group

# connect synapses object

# set up poisson input for synapse model

# initialize synapse traces with eps

# set up monitors

# write function for generating spike train correlations (get it from lovisa)

# run simulation

# plot?

# export to .csv

# --------------------------------------------------------------------

# --------------------------------------------------------------------