from brian2 import *
import pickle
import time as tm
from scipy.optimize import curve_fit
import scipy.stats as stats
sys.path.append("./")
from brian_bcpnn.models.chrysanthidis_2025.chr_params import chr_namespace, chr_equations
from brian_bcpnn.models.tully_2014.tully_params import tully_namespace, tully_equations

import brian_bcpnn.utils.synapse_utils as syls
from brian_bcpnn.plot.synapses import plot_connectivity

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
        self.REC_TRACES = ['Z_fast', 'E_fast', 'P_fast']
        self.S_REC_TRACES = ['E_syn', 'P_syn']
        self.REC_TRACES_NMDA = ['Z_slow', 'E_slow', 'P_slow']

        self.init_rec(eqs, filepath)

        self.init_s_rec(eqs, filepath)

        if N_BA > 0:
            self.init_ba(eqs)

        # RECURRENT LAYER POISSON INPUT
        self.init_poisson()

    def init_rec(self, eqs, filepath, b_slow=False):
        # RECURRENT HYPER-MINI-COLUMN LAYER
        self.REC = NeuronGroup(
            self.N, model=eqs['eqs_rec'], method='euler', threshold=eqs['threshold_rec'], reset=eqs['reset_rec'], refractory=eqs['refractory_rec']
        )
        self.network.add(self.REC)

        if filepath is not None:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.REC.V_m = data['V_m']*mV
                # I_w TODO compare if this is necessary
                self.REC.I_w = data['I_w']*nA
                self.REC.Z_fast = data['Z_fast']
                self.REC.E_fast = data['E_fast']
                self.REC.P_fast = data['P_fast']

                if b_slow:
                    self.REC.Z_slow = data['Z_slow']
                    self.REC.E_slow = data['E_slow']
                    self.REC.P_slow = data['P_slow']
        else:
            self.REC.V_m[:] = np.random.uniform(-100., -60., size=(self.N,)) * mV

    def init_s_rec(self, eqs, filepath):
        # RECURRENT LAYER BCPNN SYNAPSES
        # TODO sample i-j distances + synaptic delay
        self.S_REC = Synapses(
            self.REC, self.REC, model=eqs['bcpnn_syn_model'], on_pre=eqs['bcpnn_syn_on_pre'], method='euler', delay=self.namespace['t_delay']
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

                self.S_REC.E_syn = data['E_syn_AMPA']
                self.S_REC.P_syn = data['P_syn_AMPA']
        else:
            p_c = self.n_inc_con/(self.N_H*self.N_pyr) # assuming number of active MC per HC per pattern = 1
            if self.verbose:
                print(f'Randomly generating network connectivity with p_c = {round(p_c, 2)}')

            source_rec, target_rec = syls.get_rec_synapses(
                self.N_H, self.N_M, self.N_pyr, 
                # 0.9, 0.5, 0.1 # to test connectivity
                cp_same_mini=p_c,
                cp_same_hyper=p_c,
                cp_diff_hyper=p_c
            )
            self.S_REC.connect(i=source_rec, j=target_rec)
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
            self.REC, self.BA, on_pre=eqs['pyr_basket_on_pre'], method='euler', delay=self.namespace['t_delay']
        )
        # BA --> PYR
        self.S_BP = Synapses(
            self.BA, self.REC, on_pre=eqs['basket_pyr_on_pre'], method='euler', delay=self.namespace['t_delay']
        )
        self.S_BP.connect(i=sB, j=tP)
        self.S_PB.connect(i=sP, j=tB)
        self.network.add([self.S_PB, self.S_BP])

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

    def run(self, time, namespace=None):
        self.check_inits()
        current_time = tm.time()
        if namespace is not None:
            self.network.run(time, namespace=namespace)
        elif self.namespace is not None:
            self.network.run(time, namespace=self.namespace)
        else:
            self.network.run(time)
        current_time = tm.time() - current_time
        if self.verbose: 
            print(f'Total simulation time = {round(current_time, 2)} seconds.')

    def init_traces(self, filepath=None, S_NMDA=None):
        eps = self.namespace['eps']
        if self.verbose:
            print(f'Initialising model traces with eps={eps}')
        
        synapse_list = [self.S_REC]
        if S_NMDA is not None:
            synapse_list.append(S_NMDA)

        mode_list = ['fast', 'slow']
        for mode in mode_list:
            self.REC.set_states({f'Z_{mode}': eps, f'E_{mode}': eps, f'P_{mode}': eps})
        for synapse in synapse_list:
            synapse.set_states({f'E_syn': eps**2, f'P_syn': eps**2})
        
        if filepath is None:
            return
        
        data = None
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        for synapse, mode in zip(synapse_list, mode_list):
            fast_mode = mode == 'fast'

            # get P trace distribution (incl mean and std)
            p_data = data[f'P_{mode}']
            p_mean = mean(p_data)
            p_std = std(p_data)

            # set REC P trace (of given mode) equal to uniformly sampled array
            normal_p_samples = np.random.normal(p_mean, p_std, size=(self.N,))
            if fast_mode:
                self.REC.P_fast[:] = normal_p_samples
            else:
                self.REC.P_slow[:] = normal_p_samples

            # loop over zipped pre and post synapse indices
            sampled_weights = np.random.normal(0, 0.1, size=(len(synapse.i)))
            for i_syn, (n_pre, n_post) in enumerate(zip(synapse.i, synapse.j)):

                # sample weight as a normal of N(0, 1)
                current_weight = sampled_weights[i_syn]

                # calculate pre-post P trace product
                p_pre = self.REC.P_fast[n_pre] if fast_mode else self.REC.P_slow[n_pre]
                p_post = self.REC.P_fast[n_post] if fast_mode else self.REC.P_slow[n_post]

                # calculate corresponding synaptic P trace and overwrite current P syn trace
                synapse.P_syn[i_syn] = (10**current_weight) * p_pre * p_post



    def save_traces(self, path, S_NMDA=None):
        data = dict()

        print(S_NMDA is None)

        
        # TODO optimize code somehow in this way:
        # for trace in ['Z', 'E', 'P']:
        #     for mode in ['fast', 'slow']:
        #         var_string = f'{trace}_{mode}'
        #         data[var_string] = self.REC.get_states([var_string], units=False)
        data['Z_fast'] = self.REC.Z_fast/1
        data['E_fast'] = self.REC.E_fast/1
        data['P_fast'] = self.REC.P_fast/1
        data['Z_slow'] = self.REC.Z_slow/1
        data['E_slow'] = self.REC.E_slow/1
        data['P_slow'] = self.REC.P_slow/1

        synapse_list = [self.S_REC]
        if S_NMDA is not None:
            synapse_list.append(S_NMDA)

        for synapse, mode in zip(synapse_list, ['fast', 'slow']):
            data[f'E_syn_{mode}'] = synapse.E_syn/1
            data[f'P_syn_{mode}'] = synapse.P_syn/1

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
            namespace=chr_namespace, eqs=chr_equations
    ):
        super().__init__(
            N_H=N_H, N_M=N_M, N_pyr=N_pyr, N_BA=N_BA,
            namespace=namespace, eqs=eqs
        )

        self.S_NMDA = None

    # @Override
    def init_rec(self, eqs, filepath):
        return super().init_rec(eqs, filepath, b_slow=True)

    # @Override
    def init_s_rec(self, eqs, filepath):

        # FAST / AMPA
        self.S_REC = Synapses(
            self.REC, self.REC, model=eqs['fast_syn_model'], on_pre=eqs['fast_syn_on_pre'], method='euler', delay=self.namespace['t_delay']
        )
        self.network.add(self.S_REC)

        # SLOW / NMDA
        self.S_NMDA = Synapses(
            self.REC, self.REC, model=eqs['slow_syn_model'], on_pre=eqs['slow_syn_on_pre'], method='euler', delay=self.namespace['t_delay']
        )
        self.network.add(self.S_NMDA)

        # INTER-MC
        self.S_MC = Synapses(
            self.REC, self.REC, model=eqs['inter_mc_model'], on_pre=eqs['inter_mc_on_pre'], method='euler', delay=self.namespace['t_delay']
        )
        self.network.add(self.S_MC)

        # create connections
        if filepath is not None:
            # TODO this is deprecated if statistic initialisation works well
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                if self.verbose:
                    print(f'Initialising model parameters from file {filepath}')
                source_rec = data['S_source']
                target_rec = data['S_target']
                self.S_REC.connect(i=source_rec, j=target_rec)
                self.S_NMDA.connect(i=source_rec, j=target_rec)

                for synapse, mode in zip([self.S_REC, self.S_NMDA], ['AMPA', 'NMDA']):
                    synapse.E_syn = data[f'E_syn_{mode}']
                    synapse.P_syn = data[f'P_syn_{mode}']
        else:
            p_c = self.n_inc_con/(self.N_H*self.N_pyr) # assuming number of active MC per HC per pattern = 1
            if self.verbose:
                print(f'Randomly generating network connectivity with p_c = {round(p_c, 2)}')

            source_rec, target_rec = syls.get_rec_synapses(
                self.N_H, self.N_M, self.N_pyr, 
                cp_same_mini=0,
                cp_same_hyper=p_c,
                cp_diff_hyper=p_c
            )
            self.S_REC.connect(i=source_rec, j=target_rec)
            self.S_NMDA.connect(i=source_rec, j=target_rec)

            source_inter, target_inter = syls.get_rec_synapses(
                self.N_H, self.N_M, self.N_pyr,
                cp_same_mini=p_c,
                cp_same_hyper=0,
                cp_diff_hyper=0
            )
            self.S_MC.connect(i=source_inter, j=target_inter)

            # _, [ax1, ax2, ax3] = plt.subplots(1, 3)
            # plot_connectivity(ax1, self.S_REC, self.N)
            # plot_connectivity(ax2, self.S_NMDA, self.N)
            # plot_connectivity(ax3, self.S_MC, self.N)
            # plt.show()
            
            self.init_traces(S_NMDA=self.S_NMDA)

    # @Override
    def init_traces(self, filepath=None, S_NMDA=None):
        super().init_traces(filepath=filepath, S_NMDA=self.S_NMDA)

    # @Override
    def save_traces(self, path):
        return super().save_traces(path, self.S_NMDA)