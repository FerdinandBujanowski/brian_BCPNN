# RECURRENT HYPER-MINI-COLUMN LAYER
eqs_rec = '''
# VOLTAGE ----------------------------------------
dV_m/dt = (
    + g_L*(V_m-E_L) 
    - g_L*delta_T*exp((V_m-V_t)/delta_T)
    + I_w
    + I_beta
    + I_syn
    + I_stim
    + I_noise
)/-C_m : volt (unless refractory)
dI_w/dt = -I_w/tau_Iw : amp # adaptation current

# SYNAPTIC CURRENTS ------------------------------
g_AMPA : siemens # SUM OVER ALL SYNAPSES
I_AMPA = g_AMPA * (V_m - E_AMPA) : amp
g_NMDA : siemens # SUM OVER ALL SYNAPSES
I_NMDA = g_NMDA * (V_m - E_NMDA) : amp 
g_GABA : siemens # SUM OVER ALL SYNAPSES
dg_BA/dt = -g_BA/tau_GABA : siemens # inh current from basket cells
I_GABA = (g_GABA+g_BA) * (V_m - E_GABA) : amp 
I_syn = I_AMPA + I_NMDA + I_GABA : amp

# BETA CURRENT -----------------------------------
beta = log(clip(P_j, eps, inf)) : 1
I_beta = beta_gain * beta : amp

# EXTERNAL CURRENT -------------------------------
b_on : 1 # boolean gate of conductance based stimulation
dg_stim/dt = -g_stim/tau_AMPA : siemens
I_stim = (b_on + stim_ta(t,int(i//N_pyr))) * g_stim * (V_m-E_AMPA) : amp

# NOISE CURRENT ----------------------------------
dg_bg/dt = -g_bg/tau_AMPA : siemens
I_noise = g_bg*(V_m-E_AMPA) : amp

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
clip_p_ratio = clip(P_syn, eps**2, inf)/clip(P_i*P_j_post, eps**2, inf) : 1 (constant over dt)
w = (1-w_init)*log(clip_p_ratio) : 1 (constant over dt)
dw_init/dt = -w_init/tau_init : 1 (clock-driven)

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

# BASKET CELL EQUATIONS
eqs_basket = '''
dV_m/dt = (
    + g_L*(V_m-E_L_BA) 
    # - g_L*delta_T*exp((V_m-V_t)/delta_T)
    # + I_w
    + I_syn
)/-C_m : volt (unless refractory)
dg_ex/dt = -g_ex/tau_AMPA : siemens
I_syn = g_ex*(V_m-E_AMPA) : amp
'''
# eqs_basket_adaptive = '''
# dV_m/dt = (
#     + g_L*(V_m-E_L_BA) 
#     - g_L*delta_T*exp((V_m-V_t)/delta_T)
#     + I_w
#     + I_syn
# )/-C_m : volt (unless refractory)
# dI_w/dt = -I_w/tau_Iw : amp # adaptation current
# dg_ex/dt = -g_ex/tau_AMPA : siemens
# I_syn = g_ex*(V_m-E_AMPA) : amp
# '''
reset_ba = '''
V_m = V_r
'''
# reset_ba_adaptive = '''
# V_m = V_r
# I_w += b
# '''
pyr_basket_on_pre = '''
g_ex_post+=g_PB
'''
basket_pyr_on_pre = '''
g_BA_post+=g_BP
'''