from brian2 import *

chr_namespace = {
# SIMULATION PARAMETERS
't_sim': 0.1 * ms,
'min_num': 10e-10,
'dI': 1 * nA,

# NEURON MODEL PARAMETERS
'b': 86 * pA, # Adaptation current
'tau_Iw': 280 * ms, # Adaptation decay time constant
'C_m': 280 * pF, # Membrane capacitance
'E_L': -70 * mV, # Leak reversal potential
'g_L': 14 * nS, # Leak conductance
'delta_T': 3 * mV, # Upstroke slope factor
'V_t': -55 * mV, # Spike threshold
'V_r': -60 * mV, # Spike reset potential
'tau_ref': 5 * ms, # Refactory period
't_spike': 0.1 * ms, # time of one spike

# RECEPTOR PARAMETERS
'tau_AMPA': 5 * ms, # AMPA synaptic time cotant
'tau_NMDA': 100 * ms, # NMDA synaptic time constant
'tau_GABA': 5 * ms, # GABA synaptic time constant
'E_AMPA': 0 * mV, # AMPA reversal potential
'E_NMDA': 0 * mV, # NMDA reversal potential
'E_GABA': -75 * mV, # GABA reversal potential

# BCPNN PARAMETERS
'w_gain_AMPA': 0.33 * nS, # BCPNN AMPA gain
'w_gain_NMDA': 0.03 * nS, # BCPNN NMDA gain
'w_gain_GABA': 0.33 * nS, # BCPNN GABA gain
'beta_gain': 40 * pA, # BCPNN bias current gain # TODO set this back to >0
'f_min': 0.2 * Hz, # BCPNN lowest spiking rate
'f_max': 25 * Hz, # BCPNN highest spiking rate
'eps': 0.0026, # BCPNN lowest probability
'tau_z_i': 5 * ms, # AMPA Z trace time constant ('tau_z_AMPA')
'tau_z_j': 100 * ms, # NMDA Z trace time constant ('tau_z_NMDA')
'tau_p': 3 * second, # P trace time constant
'tau_e': 500 * ms, # E trace time constant
# 'K_normal': 0.3, # Regular plasticity
'K': 1, # Modulated plasticity ('K_reward')
't_delay': 1 * ms, # TODO paper sets one value for each synapse

# SHORT-TERM PLASTICITY PARAMETER
'U': 0.2, # Utilization factor
'tau_A': 5 * ms, # Augmentation decay time constant
'tau_D': 280 * ms, # Depression decay time constant

# CONNECTIVITY
# ...
'g_PB': 3 * nS, # pyramidal-basket connection conductance
'g_BP': -7 * nS, # basket-pyramidal connection conductance

# STIMULATION
'r_bg': 470 * Hz, # Background noise
'gr_bg': 1.5 * nS, # Background conductance (+)
'gr_bg_n': -1.5 * nS, # Background conductance (-)
'tau_bg': 5 * ms,
'V_bg': 10 * mV, # background voltage change
't_stim': 250 * ms, # Stimulation duration
'r_stim': 340 * Hz, # Stimulation rate
'gr_stim': 1.5 * nS, # Stimulation conductance
'tau_stim': 5 * ms,
'V_stim': 20 * mV,
'T_stim': 200 * ms # Interstimulus interval
}