from brian2 import *
import sys
sys.path.append("./")

from brian_bcpnn.models.bujanowski_2026.parameters import * # change this to my own param file asp
import brian_bcpnn.helper as hlp

class RecurrentLIF():

    def __init__(self, N_hyper, N_mini, complex_synmon=False):
        self.N_hyper = N_hyper
        self.N_mini = N_mini
        self.N_total = N_hyper * N_mini

        # RECURRENT HYPER-MINI-COLUMN LAYER
        eqs_rec = f'''
        # SIM PARAMS
        {SIM_PARAMS}\n
        # NEURON PARAMS
        {NEURON_PARAMS}\n

        # POSTSYNAPTIC (j) TRACES
        {TRACE_PARAMS}\n
        dS_j/dt = -S_j/sim_dt : 1
        dZ_j/dt = (S_j/(f_max*t_spike) - Z_j + epsilon*second)/tau_z : 1
        dE_j/dt = (Z_j - E_j)/tau_e : 1
        dP_j/dt = (K*(E_j-P_j))/tau_p : 1

        # bias
        beta = log(clip(P_j, min_num, inf)) : 1
        I_beta = phi*beta : amp

        # on-switch
        b_on : 1 # no stimulation protocol here
        I_on = b_on * dI : amp

        # total voltage
        g_ex : siemens # summed over all excitatory synapses
        g_inh : siemens # summed over all inhibitory synapses
        g_lat_inh : siemens # summed over all LATERAL inhibitory synapses
        dV/dt = (g_L*(V-E_L)+g_ex*(V-E_ex)+(g_inh+g_lat_inh)*(V-E_inh)+I_beta+I_on)/-C_m : volt (unless refractory)
        '''

        self.REC = NeuronGroup(self.N_total, model=eqs_rec, method='euler', threshold='V>=V_th', reset='V=V_res', refractory='t_ref')
        self.REC.set_states(SIM_PARAM_INIT)
        self.REC.set_states(NEURON_PARAM_INIT)
        self.REC.set_states(TRACE_PARAM_INIT)
        self.REC.V = -70 * mV

        # BCPNN SYNAPSES
        rec_syn_model = f'''

        # excitatory/inhibitory conductance
        {SYNAPSE_PARAMS}\n
        b_ex : 1 # 1 if synapse is excitatory, 0 if inhibitory
        w_g = w * g_max : siemens # weighted maximum condunctance

        dS_ex/dt = -S_ex/tau_ex : 1 (clock-driven) # excitatory conducting window
        # simplification to the Tully model: no alpha shaped conductance
        dalpha_ex/dt = (S_ex-alpha_ex)/tau_ex : 1  (clock-driven)
        g_ex_post = b_ex * w_g * alpha_ex : siemens (summed)

        dS_inh/dt = -S_inh/tau_inh : 1 (clock-driven) # inhibitory conducting window
        g_inh_post = (1-b_ex) * w_g * S_inh : siemens (summed)

        # PRESYNAPTIC (i) TRACES
        dS_i/dt = -S_i/sim_dt : 1 (clock-driven)
        dZ_i/dt = (S_i/(f_max*t_spike) - Z_i + epsilon*second)/tau_z : 1 (clock-driven)
        dE_i/dt = (Z_i - E_i)/tau_e : 1 (clock-driven)
        dP_i/dt = (K*(E_i-P_i))/tau_p : 1 (clock-driven)

        # BCPNN synapse
        dE_syn/dt = (Z_i*Z_j_post - E_syn)/tau_e : 1 (clock-driven)
        dP_syn/dt = (K*(E_syn-P_syn))/tau_p : 1 (clock-driven)
        w : 1 #= log(clip(P_syn, min_num, inf)/clip(P_i*P_j_post, min_num, inf)) : 1 (constant over dt)
        '''

        rec_syn_on_pre = '''
        S_i = 1
        S_ex = 1
        S_inh = 1
        '''
        rec_syn_on_post = '''
        S_j_post = 1
        '''

        self.S_REC = Synapses(self.REC, self.REC, model=rec_syn_model, on_pre=rec_syn_on_pre, on_post=rec_syn_on_post, method='euler', delay=SYNAPSE_PARAM_INIT['d'])
        self.S_REC.connect(condition='i!=j') # fully connected recurrent layer
        
        self.S_REC.set_states(SYNAPSE_PARAM_INIT)
        self.S_REC.b_ex[:] = 1 # = 'rand() > 0.2'

        # LATERAL INHIBITORY SYNAPSES
        lat_inh_model = '''
        dS_inh/dt = -S_inh/(2*ms) : 1 (clock-driven)
        g_lat_inh_post = 20 * nS * S_inh : siemens (summed)
        '''
        lat_inh_on_pre = '''
        S_inh = 1
        '''

        self.S_LAT = Synapses(self.REC, self.REC, model=lat_inh_model, on_pre=lat_inh_on_pre, method='euler')
        source_inh, target_inh = hlp.get_inh_synapses(N_hyper, N_mini)
        self.S_LAT.connect(i=source_inh, j=target_inh)

        # MONITORS
        self.spikemon = SpikeMonitor(self.REC)
        self.rec_statemon = hlp.get_BCPNN_statemon(self.REC, record=True)
        if complex_synmon:
            self.bcpnn_synmon = hlp.get_BCPNN_synmon(self.S_REC, record=True)
        else:
            self.bcpnn_synmon = hlp.get_BCPNN_weight_synmon(self.S_REC, record=True) # minimal synmon: only w tracked

        # NETWORK
        self.net = Network()
        self.net.add([self.REC, self.S_REC, self.S_LAT])
        self.monitors = [self.spikemon, self.rec_statemon, self.bcpnn_synmon]
        self.net.add(self.monitors)