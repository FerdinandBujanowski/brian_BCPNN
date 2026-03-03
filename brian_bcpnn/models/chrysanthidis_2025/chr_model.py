from brian2 import *

import sys
sys.path.append("./")

from brian_bcpnn.models.chrysanthidis_2025.chr_params import chr_namespace

N_hyper = 1
N_mini = 2
N_total = N_hyper * N_mini

# RECURRENT HYPER-MINI-COLUMN LAYER
eqs_rec = '''
# VOLTAGE ----------------------------------------
dV_m/dt = (
    - g_L*(V_m-E_L) 
    + g_L*delta_T*exp((V_m-V_t)/delta_T)
    - I_w
    + I_beta
    - I_syn
    + I_ext
    - I_noise
)/C_m : volt (unless refractory)
dI_w/dt = -I_w/tau_Iw : amp # adaptation current

# SYNAPTIC CURRENTS ------------------------------
g_AMPA : siemens # SUM OVER ALL SYNAPSES
I_AMPA = g_AMPA * (V_m - E_AMPA) : amp
g_NMDA : siemens # SUM OVER ALL SYNAPSES
I_NMDA = g_NMDA * (V_m - E_NMDA) : amp 
g_GABA : siemens # SUM OVER ALL SYNAPSES
I_GABA = g_GABA * (V_m - E_GABA) : amp 
I_syn = I_AMPA + I_NMDA + I_GABA : amp

# BETA CURRENT -----------------------------------
I_beta = beta_gain * log(clip(P_j, eps, inf)) : amp

# EXTERNAL CURRENT -------------------------------
b_cur : 1 # boolean gate of constant current stimulation
I_stim_cur = b_cur * -dI : amp

b_cond : 1 # boolean gate of conductance based stimulation
dg_stim/dt = -g_stim/tau_stim : siemens
I_stim = b_cond * g_stim * V_stim : amp
I_ext = I_stim_cur + I_stim : amp

# NOISE CURRENT ----------------------------------
dg_bg/dt = -g_bg/tau_bg : siemens
I_noise = g_bg*V_bg : amp

# SPIKE TRAIN ------------------------------------
dS/dt = -S/t_sim : 1

# POSTSYNAPTIC (j) TRACES ------------------------
dZ_j/dt = (S/(f_max*t_spike) - Z_j + eps)/tau_z_j : 1
dE_j/dt = (Z_j-E_j)/tau_e : 1
dP_j/dt = K*(E_j-P_j)/tau_p : 1
'''

reset_rec = '''
V_m = V_r
I_w += b
S = 1
'''

REC = NeuronGroup(N_total, model=eqs_rec, method='euler', threshold='V_m>V_t', reset=reset_rec, refractory='tau_ref')
REC.V_m = chr_namespace['E_L']
REC.P_j = chr_namespace['eps']


# BCPNN SYNAPSES
bcpnn_syn_model = '''
# PRESYNAPTIC (i) TRACES -------------------------
dS_i/dt = -S_i/t_sim : 1 (clock-driven)
dZ_i/dt = (S_i/(f_max*t_spike) - Z_i + eps)/tau_z_i : 1 (clock-driven)
dE_i/dt = (Z_i-E_i)/tau_e : 1 (clock-driven)
dP_i/dt = K*(E_i-P_i)/tau_p : 1 (clock-driven)

# SYNAPTIC TRACES & WEIGHTS ----------------------
dE_syn/dt = (Z_i*Z_j_post-E_syn)/tau_e : 1 (clock-driven)
dP_syn/dt = K*(E_i*E_j_post-P_syn)/tau_p : 1 (clock-driven)
clip_p_ratio = clip(P_syn/clip(P_i*P_j_post, 10e-6, inf), 10e-6, inf) : 1 (constant over dt)
w = log(clip_p_ratio) : 1 (constant over dt)

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

bcpnn_syn_on_pre = '''
S_i = 1
H_AMPA = 1
H_NMDA = 1
H_GABA = 1
'''

# TODO sample i-j distances + synaptic delay
S_REC = Synapses(REC, REC, model=bcpnn_syn_model, on_pre=bcpnn_syn_on_pre, method='euler', delay=1*ms)
S_REC.connect(condition='i!=j')
S_REC.P_i = chr_namespace['eps']
# S_REC.P_syn = chr_namespace['eps']

# POISSON INPUTS
noise_pos_input = PoissonInput(target=REC, target_var='g_bg', N=1, rate=chr_namespace['r_bg'], weight=chr_namespace['gr_bg'])
noise_neg_input = PoissonInput(target=REC, target_var='g_bg', N=1, rate=chr_namespace['r_bg'], weight=chr_namespace['gr_bg_n'])
stim_input = PoissonInput(target=REC, target_var='g_stim', N=10, rate=chr_namespace['r_stim'], weight=chr_namespace['gr_stim'])

# MONITORS
spikemon = SpikeMonitor(REC)
rec_statemon = StateMonitor(
    REC, ['V_m', 'Z_j', 'E_j', 'P_j', 'I_w', 'g_AMPA', 'I_AMPA', 'g_NMDA', 'I_NMDA', 'g_GABA', 'I_GABA', 'I_ext', 'I_beta', 'g_stim'],
    record=True
    )
bcpnn_synmon = StateMonitor(
    S_REC, ['Z_i', 'E_i', 'P_i', 'E_syn', 'P_syn', 'w', 'b_glut', 'clip_p_ratio'],
    record=True
)

# NETWORK
network = Network()
network.add([REC, S_REC])
network.add([noise_pos_input, noise_neg_input, stim_input])
network.add([spikemon, rec_statemon, bcpnn_synmon])