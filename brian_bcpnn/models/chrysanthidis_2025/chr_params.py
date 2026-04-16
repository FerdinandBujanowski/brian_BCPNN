from brian2 import *

chr_namespace = {
# SIMULATION PARAMETERS
't_sim': 0.1 * ms,

# NEURON MODEL PARAMETERS
'b': 86 * pA, # Adaptation current # 86 * pA
'tau_Iw': 280 * ms, # Adaptation decay time constant
'C_m': 280 * pF, # Membrane capacitance
'E_L': -70 * mV, # Leak reversal potential
'g_L': 14 * nS, # Leak conductance
'delta_T': 3 * mV, # Upstroke slope factor
'V_t': -55 * mV, # Spike threshold
'V_peak': 20 * mV, # peak voltage at which reset actually happens
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
'w_gain_AMPA': 0.33 * nS, # BCPNN AMPA gain # 0.33 * nS
'w_gain_NMDA': 0.0 * nS, # BCPNN NMDA gain # 0.03 * nS
'w_gain_GABA': 0.33 * nS, # BCPNN GABA gain
'beta_gain': 40 * pA, # BCPNN bias current gain
'f_min': 0.2 * Hz, # BCPNN lowest spiking rate
'f_max': 25 * Hz, # BCPNN highest spiking rate # 25 * Hz
'eps': 0.0026, # BCPNN lowest probability
'tau_z_fast': 5 * ms, # AMPA Z trace time constant ('tau_z_AMPA')
'tau_z_slow': 100 * ms, # NMDA Z trace time constant ('tau_z_NMDA')
'tau_p': 3 * second, # P trace time constant
'tau_e': 500 * ms, # E trace time constant
# 'K_normal': 0.3, # Regular plasticity
'K': 1, # Modulated plasticity ('K_reward')
'tau_init': 7.5 * second, # ease into synaptic connection at simulation beginning
't_delay': 1.5 * ms, # TODO paper sets one value for each synapse

# CONNECTIVITY
# 'cp_PP': 0.2, # pyr-pyr recurrent connection probability
# 'cp_PPL': 0.2, # pyr-pyr long-range connection probability
'cp_PB': 0.7, # pyr-basket connection probability # 0.7
'cp_BP': 0.7, # basket-pyr connection probability # 0.7
'E_L_BA': -70*mV, # basket cell leak reversal potential (added myself)
'g_PB': 3 * nS, # EXC pyramidal-basket connection conductance # 5 * nS
'g_BP': 7 * nS, # INH basket-pyramidal connection conductance
'w_inter_mc': 1, # Inter-MC connection strength

# STIMULATION
'r_bg': 400 * Hz, # Background noise # 470 Hz
'gr_bg': 1.5 * nS, # Background conductance (+)
'gr_bg_n': 1.5 * nS, # Background conductance (-)
't_stim': 250 * ms, # Stimulation duration
'r_stim': 340 * Hz, # Stimulation rate
'gr_stim': 1.5 * nS, # Stimulation conductance
'T_stim': 200 * ms, # Interstimulus interval
't_isi': 200 * ms
}

# EQUATIONS

chr_equations = {
    # RECURRENT HYPER-MINI-COLUMN LAYER
    'eqs_rec': '''
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
    g_AMPA : siemens # SUM OVER ALL FAST SYNAPSES
    g_MC_AMPA : siemens # SUM OVER ALL INTER-MC SYNAPSES
    I_AMPA = (g_AMPA + g_MC_AMPA) * (V_m - E_AMPA) : amp
    # ------------------------------------------------
    g_NMDA : siemens # SUM OVER ALL SLOW SYNAPSES
    g_MC_NMDA : siemens # SUM OVER ALL INTER-MC SYNAPSES
    I_NMDA = (g_NMDA + g_MC_NMDA) * (V_m - E_NMDA) : amp 
    # ------------------------------------------------
    g_GABA : siemens # SUM OVER ALL FAST SYNAPSES
    dg_BA/dt = -g_BA/tau_GABA : siemens
    I_GABA = (g_GABA+g_BA) * (V_m - E_GABA) : amp 
    # ------------------------------------------------
    I_syn = I_AMPA + I_NMDA + I_GABA : amp

    # BETA CURRENT -----------------------------------
    beta_fast = log(P_fast) : 1
    beta_slow = log(P_slow) : 1
    I_beta = beta_gain * (beta_fast + beta_slow) : amp

    # EXTERNAL CURRENT -------------------------------
    b_on : 1 # boolean gate of conductance based stimulation
    dg_stim/dt = -g_stim/tau_AMPA : siemens
    I_stim = (b_on + stim_ta(t,int(i//N_pyr))) * g_stim * (V_m-E_AMPA) : amp

    # NOISE CURRENT ----------------------------------
    # TODO re-establish alpha synapses?
    dg_pos_noise/dt = -g_pos_noise/tau_AMPA : siemens
    dg_neg_noise/dt = -g_neg_noise/tau_AMPA : siemens
    I_noise = (g_pos_noise-g_neg_noise)*(V_m-E_AMPA) : amp

    # SPIKE TRAIN ------------------------------------
    dS/dt = -S/t_sim : 1

    # AMPA TRACES ------------------------------------
    dZ_fast/dt = (S/(f_max*t_spike) - Z_fast + eps)/tau_z_fast : 1
    dE_fast/dt = (Z_fast-E_fast)/tau_e : 1
    dP_fast/dt = K*(E_fast-P_fast)/tau_p : 1

    # NMDA TRACES ------------------------------------
    dZ_slow/dt = (S/(f_max*t_spike) - Z_slow + eps)/tau_z_slow : 1
    dE_slow/dt = (Z_slow-E_slow)/tau_e : 1
    dP_slow/dt = K*(E_slow-P_slow)/tau_p : 1
    ''',

    'reset_rec': '''
    V_m = V_r
    I_w += b
    S = 1
    ''',
    'threshold_rec': 'V_m>V_peak',
    'refractory_rec': 'tau_ref',

    # BCPNN SYNAPSES ---------------------------------

    # COMPLETE MODEL ---------------------------------
    'bcpnn_syn_model': '''

    # SYNAPTIC TRACES & WEIGHTS ----------------------
    dE_syn/dt = (Z_fast_pre*Z_fast_post-E_syn)/tau_e : 1 (clock-driven)
    dP_syn/dt = K*(E_syn-P_syn)/tau_p : 1 (clock-driven)
    w = (1-w_init)*log(P_syn/(P_fast_pre*P_fast_post)) : 1 (constant over dt)
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
    ''',
    'bcpnn_syn_on_pre': '''
    H_AMPA = 1
    H_NMDA = 1
    H_GABA = 1
    ''',

    # FAST SYNAPSE MODEL -----------------------------
    'fast_syn_model': '''

    # SYNAPTIC TRACES & WEIGHTS ----------------------
    dE_syn/dt = (Z_fast_pre*Z_fast_post-E_syn)/tau_e : 1 (clock-driven)
    dP_syn/dt = K*(E_syn-P_syn)/tau_p : 1 (clock-driven)
    w = (1-w_init)*log(P_syn/(P_fast_pre*P_fast_post)) : 1 (constant over dt)
    dw_init/dt = -w_init/tau_init : 1 (clock-driven)

    # CONDUCTANCES -----------------------------------
    b_glut = int(w > 0) : 1
    # AMPA -------------------------------------------
    w_AMPA = b_glut * w_gain_AMPA * w : siemens
    dH_AMPA/dt = -H_AMPA/tau_AMPA : 1 (clock-driven)
    g_AMPA_post = w_AMPA * H_AMPA : siemens (summed)
    # GABA -------------------------------------------
    w_GABA = (b_glut-1) * w_gain_GABA * w : siemens
    dH_GABA/dt = -H_GABA/tau_GABA : 1 (clock-driven)
    g_GABA_post = w_GABA * H_GABA : siemens (summed)
    ''',

    'fast_syn_on_pre': '''
    H_AMPA = 1
    H_GABA = 1
    ''',

    # SLOW SYNAPSE MODEL -----------------------------
    'slow_syn_model': '''

    # SYNAPTIC TRACES & WEIGHTS ----------------------
    dE_syn/dt = (Z_slow_pre*Z_slow_post-E_syn)/tau_e : 1 (clock-driven)
    dP_syn/dt = K*(E_syn-P_syn)/tau_p : 1 (clock-driven)
    w = (1-w_init)*log(P_syn/(P_slow_pre*P_slow_post)) : 1 (constant over dt)
    dw_init/dt = -w_init/tau_init : 1 (clock-driven)

    # CONDUCTANCES -----------------------------------
    b_glut = int(w > 0) : 1
    # NMDA -------------------------------------------
    w_NMDA = b_glut * w_gain_NMDA * w : siemens
    dH_NMDA/dt = -H_NMDA/tau_NMDA : 1 (clock-driven)
    g_NMDA_post = w_NMDA * H_NMDA : siemens (summed)
    ''',

    'slow_syn_on_pre': '''
    H_NMDA = 1
    ''',

    # INTER-MINICOLUMN SYNAPSE MODEL 
    'inter_mc_model': '''
    # AMPA -------------------------------------------
    dH_AMPA/dt = -H_AMPA/tau_AMPA : 1 (clock-driven)
    g_MC_AMPA_post = w_gain_AMPA * w_inter_mc * H_AMPA : siemens (summed)
    # NMDA -------------------------------------------
    dH_NMDA/dt = -H_NMDA/tau_NMDA : 1 (clock-driven)
    g_MC_NMDA_post = w_gain_NMDA * w_inter_mc * H_NMDA : siemens (summed)
    ''',
    'inter_mc_on_pre': '''
    H_AMPA = 1
    H_NMDA = 1
    ''',

    # BASKET CELL EQUATIONS
    'eqs_basket': '''
    dV_m/dt = (
        + g_L*(V_m-E_L_BA) 
        + I_syn
    )/-C_m : volt (unless refractory)
    dg_ex/dt = -g_ex/tau_AMPA : siemens
    I_syn = g_ex*(V_m-E_AMPA) : amp
    ''',
    'reset_ba': '''
    V_m = V_r
    ''',
    'pyr_basket_on_pre': '''
    g_ex_post+=g_PB
    ''',
    'basket_pyr_on_pre': '''
    g_BA_post+=g_BP
    '''
    }