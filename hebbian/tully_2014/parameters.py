from brian2 import *

# NEURON MODEL
C_m = 250 * pF # membrane capacitance
g_L = 16.67 * nS # leak conductance
E_L = -70 * mV # leak reversal potential
E_ex = 0 * mV # excitatory reversal potential
E_inh = -75 * mV # inhibiory reversal potential
phi = 50 * pA #50 * pA # current scaling factor
V_th = -55 * mV # membrane voltage threshold
V_res = -60 * mV # membrane reset potential
t_ref = 2 * ms # refractory period
sim_dt = 0.1 * ms # time resolution

# CHANNEL MODEL
tau_z = 10 * ms # Z trace time constant
tau_e = 100 * ms # E trace time constant
tau_p = 10000 * ms # P trace time constant
f_max = 20 * Hz # highest firing rate
epsilon = 1/(f_max*tau_p) * Hz # lowest firing rate
t_spike = sim_dt # spike duration

f_on = 1000 / f_max # inter-spike time for max frequency
V_diff = V_th - V_res # amount of mV to cross to emit spike
d_V_on = V_diff/f_on # change in voltage for max frequency

# SYNAPSE MODEL
g_max = 2 * nS # peak conductance
tau_ex = 0.2 * ms # alpha rise time for excitatory input
tau_inh = 2 * ms # alpha rise time for inhibitory neurons
d = sim_dt # transmission delay
K = 1 # learning rate -> 0 to freeze plasticity