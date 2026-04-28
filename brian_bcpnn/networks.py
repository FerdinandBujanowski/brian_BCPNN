from brian2 import *
import pickle
import time as tm
from scipy.optimize import curve_fit
import scipy.stats as stats
sys.path.append("./")
from brian_bcpnn.models.chrysanthidis_2025.chr_params import chr_namespace, chr_equations
from brian_bcpnn.models.chrysanthidis_2025.fiebig_params import fiebig_equations, fiebig_namespace
from brian_bcpnn.models.tully_2014.tully_params import tully_namespace, tully_equations

import brian_bcpnn.utils.synapse_utils as syls
# from brian_bcpnn.plot.synapses import plot_connectivity

MAX_PYR = 30
MAX_BA = 4

class CorticalNetwork():

    def __init__(
            self, N_H, N_M, N_pyr=MAX_PYR, N_BA=MAX_BA, 
            namespace=None, eqs=None, filepath=None,
            n_inc_con=100, verbose=True
    ):

        self.N_H = N_H
        self.N_M = N_M
        self.N_pyr = N_pyr
        self.n_inc_con = n_inc_con
        self.N = N_H * N_M * N_pyr

        self.N_BA = N_BA
        self.N_BA_total = N_H * N_M * N_BA
        self.monitors = dict()
        self.inputs = dict()
        self.network = Network()

        self.verbose = verbose

        if namespace is not None:
            self.set_namespace(namespace)
            self.namespace['N_pyr'] = N_pyr
        self.REC_TRACES = ['Z', 'E', 'P']
        self.S_REC_TRACES = ['E_syn', 'P_syn']

        self.init_rec(eqs, filepath)

        self.init_s_rec(eqs, filepath)

        if N_BA > 0:
            self.init_ba(eqs)

        # RECURRENT LAYER POISSON INPUT
        self.init_poisson()

    def init_rec(self, eqs, filepath):
        # RECURRENT HYPER-MINI-COLUMN LAYER
        self.REC = NeuronGroup(
            self.N, model=eqs['eqs_rec'], method='euler', threshold=eqs['threshold_rec'], reset=eqs['reset_rec'], refractory=eqs['refractory_rec']
        )
        self.network.add(self.REC)

        if filepath is not None:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.REC.Z = data['Z']
                self.REC.E = data['E']
                self.REC.P = data['P']

        self.REC.V_m[:] = np.random.uniform(-75, -65, size=(self.N,)) * mV

    def init_s_rec(self, eqs, filepath):
        # RECURRENT LAYER BCPNN SYNAPSES
        # TODO sample i-j distances + synaptic delay
        self.S_REC = Synapses(
            self.REC, self.REC, model=eqs['full_syn_model'], on_pre=eqs['full_syn_on_pre'], method='euler'#, delay=self.namespace['t_delay_long']
        )
        self.network.add(self.S_REC)

        # create connections
        if filepath is not None:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                if self.verbose:
                    print(f'Initialising model parameters from file {filepath}')
                source_rec = data['S_source']
                target_rec = data['S_target']
                self.S_REC.connect(i=source_rec, j=target_rec)
                self.S_REC.delay = self.namespace['t_delay_long']

                self.S_REC.E_syn = data['E_syn']
                self.S_REC.P_syn = data['P_syn']
        else:
            self.p_c = self.n_inc_con/(self.N_H*self.N_pyr) # assuming number of active MC per HC per pattern = 1
            if self.verbose:
                print(f'Randomly generating network connectivity with p_c = {round(self.p_c, 2)}')

            source_rec, target_rec = syls.get_rec_synapses(
                self.N_H, self.N_M, self.N_pyr, 
                # 0.9, 0.5, 0.1 # to test connectivity
                cp_same_mini=self.p_c,
                cp_same_hyper=self.p_c,
                cp_diff_hyper=self.p_c
            )
            self.S_REC.connect(i=source_rec, j=target_rec)
            self.S_REC.delay = self.namespace['t_delay_long']
            self.init_traces()

    def init_ba(self, eqs):
        # BASKET CELLS
        self.BA = NeuronGroup(
            self.N_BA_total, model=eqs['eqs_basket'], method='euler', threshold='V_m>V_t', reset=eqs['reset_ba'], refractory='tau_ref'
        )
        self.network.add(self.BA)
        self.BA.V_m = self.namespace['E_L']

        # BASKET CELL SYNAPSES
        (sP, tB, sB, tP) = syls.get_basket_synapses(
            self.N_H, self.N_M, self.N_pyr, self.N_BA,
            cp_PB=self.namespace['cp_PB'],
            cp_BP=self.namespace['cp_BP'],
            symmetry=True
        )
        # PYR --> BA
        self.S_PB = Synapses(
            self.REC, self.BA, model=eqs['syn_PB'], on_pre=eqs['pyr_basket_on_pre'], method='euler'#, delay=self.namespace['t_delay']
        )
        # BA --> PYR
        self.S_BP = Synapses(
            self.BA, self.REC, model=eqs['syn_BP'], on_pre=eqs['basket_pyr_on_pre'], method='euler'#, delay=self.namespace['t_delay']
        )
        self.S_BP.connect(i=sB, j=tP)
        self.S_BP.delay = self.namespace['t_delay']
        self.S_PB.connect(i=sP, j=tB)
        self.S_PB.delay = self.namespace['t_delay']
        self.network.add([self.S_PB, self.S_BP])

        # calculate strength of basket cell conductances
        # n_inc_per_hc = self.p_c * self.N_H * (self.N_M**2) * (self.N_pyr**2)
        # basket_scalar = n_inc_per_hc / 460800
        # g_PB = 3 * basket_scalar * nS
        # g_BP = self.namespace['g_BP_scalar'] * g_PB
        # self.namespace['g_PB'] = g_PB
        # self.namespace['g_BP'] = g_BP
        # self.BA.V_m[:] = np.random.uniform(-100, -60, size=(self.N_BA_total)) * mV
        # print(f'Setting up basket cells with g_PB={round(g_PB/nS,2)}*nS and g_BP={round(g_BP/nS,2)*nS}')



    def add_monitor(self, monitor, name):
        self.monitors[name] = monitor
        self.network.add(monitor)

    def delete_monitor(self, name):
        monitor = self.monitors.pop(name)
        self.network.remove(monitor)
        del monitor

    def add_poisson(self, poisson_input, name):
        self.inputs[name] = poisson_input
        self.network.add(poisson_input)

    def init_poisson(self):
        pass

    def delete_poisson(self, name):
        poisson_input = self.inputs.pop(name)
        pass
        self.network.remove(poisson_input)
        del poisson_input

    def set_namespace(self, namespace):
        self.namespace = namespace

    def run(self, time, namespace=None, verbose=True):
        self.check_inits()
        current_time = tm.time()
        if namespace is not None:
            self.network.run(time, namespace=namespace)
        elif self.namespace is not None:
            self.network.run(time, namespace=self.namespace)
        else:
            self.network.run(time)
        current_time = tm.time() - current_time
        if verbose: 
            print(f'Total simulation time = {round(current_time, 2)} seconds.')

    def init_traces(self, model="eps", filepath=None, S_NMDA=None, baseline=1*Hz):
        MODEL_EPS = "eps"
        MODEL_FILE = "file"
        MODEL_ZERO_WEIGHT = "zero_weight"
        MODEL_PAPER = "paper"
        model_options = [MODEL_EPS, MODEL_FILE, MODEL_ZERO_WEIGHT, MODEL_PAPER]
        if model not in model_options:
            raise ValueError(f"Unknown initialisation model chosen. Options are {", ".join(model_options)}")
        
        eps = self.namespace['eps']
        if self.verbose:
            print(f'Initialising network traces with model "{model}" and eps={eps}')
    
        if model == "eps":
            self.REC.set_states({'Z': eps, 'E': eps, 'P': eps})
            self.S_REC.set_states({'E_syn': eps**2, 'P_syn': eps**2})
            return
        
        elif model == "file" and filepath is not None:
        
            data = None
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

                # get P trace distribution (incl mean and std)
                p_data = data['P']
                p_mean = mean(p_data)
                p_std = std(p_data)

                # set REC P trace (of given mode) equal to uniformly sampled array
                normal_p_samples = np.random.normal(p_mean, p_std, size=(self.N,))
                self.REC.P[:] = normal_p_samples

                # loop over zipped pre and post synapse indices
                sampled_weights = np.random.normal(0, 0.1, size=(len(self.S_REC.i)))
                for i_syn, (n_pre, n_post) in enumerate(zip(self.S_REC.i, self.S_REC.j)):

                    # sample weight as a normal of N(0, 1)
                    current_weight = sampled_weights[i_syn]

                    # calculate pre-post P trace product
                    p_pre = self.REC.P[n_pre]
                    p_post = self.REC.P[n_post]

                    # calculate corresponding synaptic P trace and overwrite current P syn trace
                    self.S_REC.P_syn[i_syn] = (10**current_weight) * p_pre * p_post


        elif model == "zero_weight":
            p_all = baseline / self.namespace['f_max']
            p_syn_all = p_all ** 2

            self.REC.set_states({'P': p_all})
            self.S_REC.set_states({'P_syn': p_syn_all})

        elif model == "paper":
            # TODO add pattern list to get weights from
            p_all = baseline / self.namespace['f_max']
            self.REC.set_states({'P': p_all})

            p_syn_all = []
            for s, t in zip(self.S_REC.i, self.S_REC.j):
                s_H, s_M = syls.get_neuron_coords(s, self.N_M, self.N_pyr)
                t_H, t_M = syls.get_neuron_coords(t, self.N_M, self.N_pyr)

                if s_H == t_H:
                    # same hypercolumn AND different minicolumn 
                    # (there are no synapses in S_REC where H==H and M==M)
                    p_syn_all.append((10**self.namespace['intra_hc_inter_mc'])*(p_all**2))
                else:
                    if s_M == t_M:
                        # different hypercolumns, coactive (assuming orthogonal patterns)
                        p_syn_all.append((10**self.namespace['inter_hc_coactive'])*(p_all**2))
                    else:
                        p_syn_all.append((10**self.namespace['inter_hc_competing'])*(p_all**2))
            self.S_REC.set_states({'P_syn': p_syn_all})


        self.REC.set_states({'Z': eps, 'E': eps})
        self.S_REC.set_states({'E_syn': eps**2})

    def save_traces(self, path):
        data = dict()
        
        data['Z'] = self.REC.Z/1
        data['E'] = self.REC.E/1
        data['P'] = self.REC.P/1

        data['S_source'] = np.array(self.S_REC.i)
        data['S_target'] = np.array(self.S_REC.j)

        data['E_syn'] = self.S_REC.E_syn/1
        data['P_syn'] = self.S_REC.P_syn/1

        with open(path, 'wb') as f:
            pickle.dump(data, f)
            print('Successfully saved model traces.')

    def check_inits(self):
        if self.namespace['stim_ta'] is None:
            raise ValueError('No stimulation TimedArray provided.')
        # TODO rewrite this for standalone mode

        # rec_trace_values = self.REC.get_states(self.REC_TRACES)
        # for key in rec_trace_values.keys():
        #     if np.any(rec_trace_values[key] == 0.):
        #         raise ValueError(f'All trace variables should be initialized to above zero: {key}={rec_trace_values[key]}')
        # s_rec_trace_values = self.S_REC.get_states(self.S_REC_TRACES)
        # for key in s_rec_trace_values.keys():
        #     if np.any(s_rec_trace_values[key] == 0.):
        #         raise ValueError(f'All trace variables should be initialized to above zero: {key}={s_rec_trace_values[key]}')

    def add_synmon(self, variables, record):
        synmon = StateMonitor(self.S_REC, variables=variables, record=record)
        self.add_monitor(synmon, synmon.name)
        return synmon
    def add_statemon(self, variables, record):
        statemon = StateMonitor(self.REC, variables=variables, record=record)
        self.add_monitor(statemon, statemon.name)
        return statemon
    def add_spikemon(self):
        spikemon = SpikeMonitor(self.REC)
        self.add_monitor(spikemon, spikemon.name)
        return spikemon
    def add_basmon(self):
        basmon = SpikeMonitor(self.BA)
        self.add_monitor(basmon, basmon.name)
        return basmon

class ChrysanthidisNetwork(CorticalNetwork):

    def __init__(
            self, N_H, N_M, N_pyr, N_BA, N_poisson=1,
            namespace=chr_namespace, eqs=chr_equations, filepath=None 
    ):
        self.N_poisson = N_poisson
        super().__init__(N_H, N_M, N_pyr, N_BA, namespace, eqs, filepath)

        # fig, [ax1, ax2] = plt.subplots(1, 2)
        # synapses.plot_connectivity(ax1, self.S_PB, self.N, self.N_BA_total)
        # ax1.set_title('PYR-BA connectivity')
        # synapses.plot_connectivity(ax2, self.S_BP, self.N_BA_total, self.N)
        # ax2.set_title('BA-PYR connectivity')
        # plt.show()

        # MONITORS
        # ... to be added in file instantiating class

    # @Override
    def init_poisson(self):
        noise_pos_input = PoissonInput(target=self.REC, target_var='g_pos_noise', N=1, rate=self.namespace['r_bg'], weight=self.namespace['gr_bg'])
        self.add_poisson(noise_pos_input, 'pos_noise')
        noise_neg_input = PoissonInput(target=self.REC, target_var='g_neg_noise', N=1, rate=self.namespace['r_bg'], weight=self.namespace['gr_bg_n'])
        self.add_poisson(noise_neg_input, 'neg_noise')
        stim_input = PoissonInput(target=self.REC, target_var='g_stim', N=self.N_poisson, rate=self.namespace['r_stim'], weight=self.namespace['gr_stim'])
        self.add_poisson(stim_input, 'stim_input')

class TullyNetwork(CorticalNetwork):

    def __init__(self, namespace=tully_namespace, eqs=tully_equations, verbose=True):
        super().__init__(2, 1, 1, 0, namespace, eqs, verbose=verbose)

    # @Override
    def init_poisson(self):
        stim_input = PoissonInput(target=self.REC, target_var='g_stim', N=self.namespace['n_ex'], rate=self.namespace['r_ex'], weight=self.namespace['w_ex'])
        self.add_poisson(stim_input, 'stim_input')
    
    # @Override
    def init_s_rec(self, eqs, _):
        # RECURRENT LAYER BCPNN SYNAPSES
        self.S_REC = Synapses(
            self.REC, self.REC, model=eqs['bcpnn_syn_model'], on_pre=eqs['bcpnn_syn_on_pre'], method='euler', delay=self.namespace['d']
        )
        self.network.add(self.S_REC)

        self.S_REC.connect(i=0, j=1)

        self.namespace['eps'] = self.namespace['epsilon']
        self.init_traces()


class TwoSynTypeNetwork(ChrysanthidisNetwork):
    # TODO override synapse init function
    def __init__(
            self, N_H, N_M, N_pyr, N_BA,
            namespace=fiebig_namespace, eqs=fiebig_equations, filepath=None
    ):
        super().__init__(
            N_H=N_H, N_M=N_M, N_pyr=N_pyr, N_BA=N_BA,
            namespace=namespace, eqs=eqs, filepath=filepath
        )

    # @Override
    def init_s_rec(self, eqs, filepath):

        # FAST / AMPA
        self.S_REC = Synapses(
            self.REC, self.REC, model=eqs['full_syn_model'], on_pre=eqs['full_syn_on_pre'], method='euler'
        )
        self.network.add(self.S_REC)

        # INTER-MC
        self.S_MC = Synapses(
            self.REC, self.REC, model=eqs['inter_mc_model'], on_pre=eqs['inter_mc_on_pre'], method='euler'#, delay=self.namespace['t_delay']
        )
        
        self.p_c = 0 if self.N_H == 1 else 90 / ((self.N_H-1) * self.N_pyr)
        source_inter, target_inter = syls.get_rec_synapses(
                self.N_H, self.N_M, self.N_pyr,
                cp_same_mini=self.namespace['p_c_intra_mc'],
                cp_same_hyper=0,
                cp_diff_hyper=0
            )
        self.S_MC.connect(i=source_inter, j=target_inter)
        self.S_MC.delay = self.namespace['t_delay']
        self.network.add(self.S_MC)

        # create connections
        if filepath is not None:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                if self.verbose:
                    print(f'Initialising model connectivity & weights from file {filepath}')
                source_rec = data['S_source']
                target_rec = data['S_target']
                self.S_REC.connect(i=source_rec, j=target_rec)
                self.S_REC.delay = self.namespace['t_delay_long']

                # TODO export / import synaptic delay from file?
                self.S_REC.E_syn = data['E_syn']
                self.S_REC.P_syn = data['P_syn']
        else:
            if self.verbose:
                print(f'Randomly generating network connectivity with p_c = {round(self.p_c, 2)}')

            if  self.N_H > 1:
                source_rec, target_rec = syls.get_rec_synapses(
                    self.N_H, self.N_M, self.N_pyr, 
                    cp_same_mini=0,
                    cp_same_hyper=0,
                    cp_diff_hyper=self.p_c
                )
                self.S_REC.connect(i=source_rec, j=target_rec)
                self.S_REC.delay = self.namespace['t_delay_long']
            else:
                # TODO make it so that model doesn't crash when it only has 1 HC
                pass

            
            # _, [ax1, ax2, ax3] = plt.subplots(1, 3)
            # plot_connectivity(ax1, self.S_REC, self.N)
            # plot_connectivity(ax2, self.S_NMDA, self.N)
            # plot_connectivity(ax3, self.S_MC, self.N)
            # plt.show()

            # CONDUCTION DELAYS
            # delays = []
            # for i, j in zip(self.S_REC.i, self.S_REC.j):
            #     if i // self.N_pyr == j // self.N_pyr:
            #         delays.append[self.namespace['t_delay']]
            #     else:
            #         delays.append(self.namespace['t_delay']*self.namespace['t_delay_factor'])
            
            # self.S_REC.delay = delays
            # self.S_NMDA.delay = delays

            self.init_traces()