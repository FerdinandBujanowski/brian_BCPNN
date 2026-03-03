import sys
sys.path.append("./")
from brian_bcpnn.networks import EquationSystem

# RECURRENT LAYER EQUATIONS
adex_voltage = '''
# VOLTAGE ----------------------------------------
dV_m/dt = (
    - g_L*(V_m-E_L) 
    + g_L*delta_T*exp((V_m-V_t)/delta_T)
    - I_w
    + I_beta
    + I_syn
    + I_ext
)/C_m : volt
dI_w/dt = -I_w/tau_Iw : amp # adaptation current
'''
syn_currents = '''
# SYNAPTIC CURRENTS ------------------------------
g_AMPA : siemens # SUM OVER ALL SYNAPSES
I_AMPA = g_AMPA * (V_m - E_AMPA) : amp
g_NMDA : siemens # SUM OVER ALL SYNAPSES
I_NMDA = g_NMDA * (V_m - E_NMDA) : amp 
g_GABA : siemens # SUM OVER ALL SYNAPSES
I_GABA = g_GABA * (V_m - E_GABA) : amp 
I_syn = I_AMPA + I_NMDA + I_GABA : amp
'''
beta_current = '''
# BETA CURRENT -----------------------------------
I_beta = beta_gain * log(clip(P_j, min_num, inf)) : amp
'''
ext_stim = '''
# EXTERNAL CURRENT -------------------------------
b_cur : 1 # boolean gate of constant current stimulation
I_stim_cur = b_cur * -dI : amp

b_cond : 1 # boolean gate of conductance based stimulation
dg_stim/dt = -g_stim/tau_stim : siemens
I_stim = b_cond * g_stim * V_stim : amp
I_ext = I_stim_cur + I_stim : amp
'''
noise_current = '''
# NOISE CURRENT ----------------------------------
dg_bg/dt = -g_bg/tau_bg : siemens
I_noise = g_bg*V_bg : amp
'''
spike_train = '''
# SPIKE TRAIN ------------------------------------
dS/dt = -S/t_sim : 1
'''
postsyn_trace = '''
# POSTSYNAPTIC (j) TRACES ------------------------
dZ_j/dt = (S/(f_max*t_spike) - Z_j + eps)/tau_z_j : 1
dE_j/dt = (Z_j-E_j)/tau_e : 1
dP_j/dt = K*(E_j-P_j)/tau_p : 1
'''

adex_reset = '''
V_m = V_r
I_w += b
'''
adex_threshold = 'V_m>V_t'
adex_refractory = 'tau_ref'

# BCPNN SYNAPSE
presyn_trace = '''
# PRESYNAPTIC (i) TRACES -------------------------
dS_i/dt = -S_i/t_sim : 1 (clock-driven)
dZ_i/dt = (S_i/(f_max*t_spike) - Z_i + eps)/tau_z_i : 1 (clock-driven)
dE_i/dt = (Z_i-E_i)/tau_e : 1 (clock-driven)
dP_i/dt = K*(E_i-P_i)/tau_p : 1 (clock-driven)
'''
syn_trace = '''
# SYNAPTIC TRACES & WEIGHTS ----------------------
dE_syn/dt = (Z_i*Z_j_post-E_syn)/tau_e : 1 (clock-driven)
dP_syn/dt = K*(E_i*E_j_post-P_syn)/tau_p : 1 (clock-driven)
clip_p_prod = clip(P_i*P_j_post, eps, inf) : 1 (constant over dt)
clip_p_ratio = clip(P_syn/clip_p_prod, eps, inf) : 1 (constant over dt)
w = log(clip_p_ratio) : 1 (constant over dt)
'''
syn_cond = '''
# CONDUCTANCES -----------------------------------
b_glut = int(w > 0) : 1
# AMPA -------------------------------------------
w_AMPA = b_glut * w_gain_AMPA * w : siemens
dH_AMPA/dt = -H_AMPA/tau_AMPA : 1 (clock-driven)
g_AMPA_post = w_AMPA * H_AMPA : siemens (summed)
# NMDA -------------------------------------------
w_NMDA = b_glut * w_gain_NMDA * w : siemens
dH_NMDA/dt = -H_NMDA/tau_NMDA : 1 (clock-driven)
g_NMDA_post = w_NMDA * H_NMDA : siemens (summed)
# GABA -------------------------------------------
w_GABA = (b_glut-1) * w_gain_GABA * w : siemens
dH_GABA/dt = -H_GABA/tau_GABA : 1 (clock-driven)
g_GABA_post = w_GABA * H_GABA : siemens (summed)
'''
syn_on_pre = '''
S_i = 1
H_AMPA = 1
H_NMDA = 1
H_GABA = 1
'''

# Equation systems
recurrent_layer_eqs = EquationSystem(
        adex_voltage
).add(  syn_currents
).add(  beta_current
).add(  ext_stim
).add(  spike_train
).add(  postsyn_trace)