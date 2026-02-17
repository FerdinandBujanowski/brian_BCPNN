from brian2 import *

# NEURON MODEL PARAMETERS
b = 86 * pA # Adaptation current
tau_lw = 280 * ms # Adaptation decay time constant
C_m = 280 * pF # Membrane capacitance
E_L = -70.6 * mV # Leak reversal potential
g_L = 14 * nS # Leak conductance
delta_T = 3 * mV # Upstroke slope factor
V_t = -55 * mV # Spike threshold
V_r = -60 * mV # Spike reset potential
tau_ref = 5 * ms # Refactory period

# RECEPTOR PARAMETERS
tau_AMPA = 5 * ms # AMPA synaptic time constant
tau_NMDA = 100 * ms # NMDA synaptic time constant
tau_GABA = 5 * ms # GABA synaptic time constant
E_AMPA = 0 * mV # AMPA reversal potential
E_NMDA = 0 * mV # NMDA reversal potential
E_GABA = -75 * mV # GABA reversal potential

# BCPNN PARAMETERS
w_gain_AMPA = 0.33 * nS # BCPNN AMPA gain
w_gain_NMDA = 0.03 * nS # BCPNN NMDA gain
beta_gain = 40 * pA # BCPNN bias current gain
f_min = 0.2 * Hz # BCPNN lowest spiking rate
f_max = 25 * Hz # BCPNN highest spiking rate
eps = 0.0026 # BCPNN lowest probability
tau_p = 30 * second # P trace time constant
tau_e = 500 * ms # E trace time constant
tau_Z_AMPA = 5 * ms # AMPA Z trace time constant
tau_Z_NMDA = 100 * ms # NMDA Z trace time constant
K_normal = 0.3 # Regular plasticity
K_reward = 1 # Modulated plasticity

# SHORT-TERM PLASTICITY PARAMETER
U = 0.2 # Utilization factor
tau_A = 5 * ms # Augmentation decay time constant
tau_D = 280 * ms # Depression decay time constant