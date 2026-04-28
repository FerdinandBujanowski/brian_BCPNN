from brian2 import *

fiebig_namespace = {
    't_sim': 0.1*ms,

    'b': 86*pA,
    'tau_Iw': 500*ms,
    'C_m': 280*pF,
    'E_L': -70*mV,
    'g_L': 14*nS,
    'delta_T': 3*mV,
    'V_t': -55*mV,
    'V_r': -80*mV,
    'tau_ref': 5*ms,
    'U': 0.33,

    'tau_rec': 500*ms,
    'tau_AMPA': 5*ms,
    'tau_NMDA': 100*ms,
    'tau_GABA': 5*ms,
    'E_AMPA': 0*mV,
    'E_NMDA': 0*mV,
    'E_GABA': -75*mV,

    'K_AMPA': 1,
    'K_NMDA': 1,

    'w_gain_AMPA': 0.1*0.78*3.93*nS,
    'w_gain_NMDA': 0.1*4*0.21*nS, #TODO increase x5... see if stabilizes
    'w_gain_GABA': 0.1*3*3.93*nS,

    'beta_gain': pA*30,

    'f_min': 0.5*Hz,
    'f_max': 50*Hz,
    'eps': 0.01,
    't_spike': 0.1*ms,

    'tau_z_fast': 5*ms,
    'tau_z_slow': 5*ms,
    'tau_e': 100*ms,
    'tau_p': 1*second,

    't_delay': 1.5*ms,
    't_delay_long': 25*ms,

    'intra_hc_intra_mc': 2.5, # FIXED 
    # 'intra_hc_inter_mc': 0, # this one won't matter if no connection between diff MCs in same HC
    'inter_hc_coactive': 2,
    'inter_hc_competing': -2, #-0.06,

    'p_c_intra_mc': 0.25,

    'r_bg': 550*Hz,
    'gr_bg': 1.5*nS,
    'gr_bg_n': 1.5*nS,
    'r_stim': 600*Hz,
    'gr_stim': 1.5*nS,

    'E_L_BA': -70*mV,
    'cp_PB': 0.7,
    'cp_BP': 0.7, 

    'g_PB_factor': 0.15,
    'g_PB': 3.5*nS,
    'g_BP_factor': 0.8,
    'g_BP': 20*nS,

    't_stim': 100*ms,
    't_isi': 300*ms
}

fiebig_equations = {
    # RECURRENT HYPER-MINI-COLUMN LAYER
    'eqs_rec': '''
    # VOLTAGE ----------------------------------------
    dV_m/dt = (
        - g_L*(V_m-E_L) 
        + g_L*delta_T*exp((V_m-V_t)/delta_T)
        - I_w
        + I_beta
        - I_syn
        - I_stim
        - I_noise
    )/C_m : volt (unless refractory)
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
    g_BA : siemens # SUM OVER ALL BASKET-PYR SYNAPSES
    I_GABA = (g_GABA + g_BA) * (V_m - E_GABA) : amp 
    # ------------------------------------------------
    I_syn = I_AMPA + I_NMDA + I_GABA : amp

    # BETA CURRENT -----------------------------------
    beta_fast = log(P_fast) : 1
    beta_slow = log(P_slow) : 1
    I_beta = beta_gain * (beta_slow) : amp

    # EXTERNAL CURRENT -------------------------------
    b_on : 1 # boolean gate of conductance based stimulation
    dg_stim/dt = -g_stim/tau_AMPA : siemens
    I_stim = (b_on + stim_ta(t,int(i//N_pyr))) * g_stim * (V_m-E_AMPA) : amp

    # NOISE CURRENT ----------------------------------
    dg_pos_noise/dt = -g_pos_noise/tau_AMPA : siemens
    dg_neg_noise/dt = -g_neg_noise/tau_AMPA : siemens
    I_noise = g_pos_noise*(V_m-E_AMPA) + g_neg_noise*(V_m-E_GABA) : amp

    # SPIKE TRAIN ------------------------------------
    dS/dt = -S/t_sim : 1

    # AMPA TRACES ------------------------------------
    dZ_fast/dt = (S/(f_max*t_spike) - Z_fast + eps)/tau_z_fast : 1
    dE_fast/dt = (Z_fast-E_fast)/tau_e : 1
    dP_fast/dt = K_AMPA*(E_fast-P_fast)/tau_p : 1

    # NMDA TRACES ------------------------------------
    dZ_slow/dt = (S/(f_max*t_spike) - Z_slow + eps)/tau_z_slow : 1
    dE_slow/dt = (Z_slow-E_slow)/tau_e : 1
    dP_slow/dt = K_NMDA*(E_slow-P_slow)/tau_p : 1
    ''',

    'reset_rec': '''
    V_m = V_r
    I_w = b
    S = 1
    ''',
    'threshold_rec': 'V_m>V_t',
    'refractory_rec': 'tau_ref',

    # BCPNN SYNAPSES ---------------------------------

    # FAST SYNAPSE MODEL -----------------------------
    'fast_syn_model': '''

    # SYNAPTIC TRACES & WEIGHTS ----------------------
    dE_syn/dt = (Z_fast_pre*Z_fast_post-E_syn)/tau_e : 1 (clock-driven)
    dP_syn/dt = K_AMPA*(E_syn - P_syn)/tau_p : 1 (clock-driven)
    w = log(P_syn/(P_fast_pre*P_fast_post)) : 1 (constant over dt)

    # CONDUCTANCES -----------------------------------
    b_glut = int(w > 0) : 1
    # AMPA -------------------------------------------
    w_AMPA = b_glut * w_gain_AMPA * w : siemens
    dH_AMPA/dt = -H_AMPA/tau_AMPA : 1 (clock-driven)
    g_AMPA_post = w_AMPA * H_AMPA * x : siemens (summed)
    # GABA -------------------------------------------
    w_GABA = (b_glut-1) * w_gain_GABA * w : siemens
    dH_GABA/dt = -H_GABA/tau_GABA : 1 (clock-driven)
    g_GABA_post = w_GABA * H_GABA * x : siemens (summed)

    # DEPLETION --------------------------------------
    dx/dt = (1-x)/tau_rec : 1 (clock-driven)
    ''',

    'fast_syn_on_pre': '''
    H_AMPA = 1
    H_GABA = 1
    x -= U * x
    ''',

    # SLOW SYNAPSE MODEL -----------------------------
    'slow_syn_model': '''

    # SYNAPTIC TRACES & WEIGHTS ----------------------
    dE_syn/dt = (Z_slow_pre*Z_slow_post-E_syn)/tau_e : 1 (clock-driven)
    dP_syn/dt = K_NMDA*(E_syn - P_syn)/tau_p : 1 (clock-driven)
    w = log(P_syn/(P_slow_pre*P_slow_post)) : 1 (constant over dt)

    # CONDUCTANCES -----------------------------------
    b_glut = int(w > 0) : 1
    # NMDA -------------------------------------------
    w_NMDA = b_glut * w_gain_NMDA * w : siemens
    dH_NMDA/dt = -H_NMDA/tau_NMDA : 1 (clock-driven)
    g_NMDA_post = w_NMDA * H_NMDA * x : siemens (summed)

    # DEPLETION --------------------------------------
    dx/dt = (1-x)/tau_rec : 1 (clock-driven)
    ''',

    'slow_syn_on_pre': '''
    H_NMDA = 1
    x -= U * x
    ''',

    # INTER-MINICOLUMN SYNAPSE MODEL 
    'inter_mc_model': '''
    # AMPA -------------------------------------------
    dH_AMPA/dt = -H_AMPA/tau_AMPA : 1 (clock-driven)
    g_MC_AMPA_post = w_gain_AMPA * intra_hc_intra_mc * H_AMPA : siemens (summed)
    # NMDA -------------------------------------------
    dH_NMDA/dt = -H_NMDA/tau_NMDA : 1 (clock-driven)
    g_MC_NMDA_post = w_gain_NMDA * intra_hc_intra_mc * H_NMDA : siemens (summed)
    ''',
    'inter_mc_on_pre': '''
    H_AMPA = 1
    H_NMDA = 1
    ''',

    # BASKET CELL EQUATIONS
    'eqs_basket': '''
    dV_m/dt = (
        - g_L*(V_m-E_L_BA) 
        - I_syn
    )/C_m : volt (unless refractory)
    g_ex : siemens # SUMMED
    # dg_ex/dt = -g_ex/tau_AMPA : siemens
    I_syn = g_ex*(V_m-E_AMPA) : amp
    ''',

    'syn_PB': '''
    dH_ex/dt = -H_ex/tau_AMPA : 1 (clock-driven)
    g_ex_post = H_ex * g_PB_factor * g_PB : siemens (summed)
    ''',
    'syn_BP': '''
    dH_BA/dt = -H_BA/tau_GABA : 1 (clock-driven)
    g_BA_post = H_BA * g_BP_factor * g_BP : siemens (summed)
    ''',
    'reset_ba': '''
    V_m = V_r
    ''',
    'pyr_basket_on_pre': '''
    H_ex = 1
    ''',
    'basket_pyr_on_pre': '''
    H_BA = 1
    '''
    }