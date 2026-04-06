from brian2 import *
import pickle
import time as tm
from scipy.optimize import curve_fit
import scipy.stats as stats
sys.path.append("./")
from brian_bcpnn.models.chrysanthidis_2025.chr_params import chr_namespace, chr_equations
from brian_bcpnn.models.tully_2014.tully_params import tully_namespace, tully_equations

import brian_bcpnn.utils.synapse_utils as syls
import brian_bcpnn.utils.stim_utils as stils
from brian_bcpnn.utils.stim_utils import Pattern, ColumnCoords

class CorticalNetwork():

    def __init__(
            self, N_H, N_M, N_pyr, N_BA, 
            namespace, eqs, filepath=None,
            n_inc_con=100
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

        if namespace is not None:
            self.set_namespace(namespace)
            self.namespace['N_pyr'] = N_pyr
        self.REC_TRACES = ['Z_j', 'E_j', 'P_j']
        self.S_REC_TRACES = ['Z_i', 'E_i', 'P_i', 'E_syn', 'P_syn']

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
                self.REC.V_m = data['V_m']*mV
                # I_w TODO compare if this is necessary
                self.REC.I_w = data['I_w']*nA
                self.REC.Z_j = data['Z_j']
                self.REC.E_j = data['E_j']
                self.REC.P_j = data['P_j']
        else:
            self.REC.V_m = self.namespace['E_L']

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
                print(f'Initialising model parameters from file {filepath}')
                source_rec = data['S_source']
                target_rec = data['S_target']
                self.S_REC.connect(i=source_rec, j=target_rec)

                self.S_REC.Z_i = data['Z_i']
                self.S_REC.E_i = data['E_i']
                self.S_REC.P_i = data['P_i']
                self.S_REC.E_syn = data['E_syn']
                self.S_REC.P_syn = data['P_syn']
        else:
            p_c = self.n_inc_con/(self.N_H*self.N_pyr) # assuming number of active MC per HC per pattern = 1
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

    # idea is to take saved params of a smaller network,
    # and to randomly initialise bigger network using statistics from saved params
    # Note that this only makes sense with networks that haven't yet learned any patterns
    def sample_params(self, filepath):
        print(f'Sampling parameter values from distributions of file {filepath}.')
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

            # reconstruct original weights
            # source_rec = data['S_source']
            target_rec = data['S_target']
            P_syn = data['P_syn']
            P_i = data['P_i']
            P_j = data['P_j']
            
            old_weights = np.zeros(shape=(len(target_rec)))
            for i, ta in enumerate(target_rec):
                old_weights[i] = log(P_syn[i]/(P_i[i]*P_j[ta]))

            # Plot distribution of weights
            w_space = np.linspace(min(old_weights), max(old_weights), 1000)
            mean_old_weights = np.mean(old_weights)
            std_old_weights = np.std(old_weights)
            print(f'mu_w={round(mean_old_weights, 3)}, std_w={round(std_old_weights, 3)}')
            sigma_old_weights = sqrt(std_old_weights)

            # plot distribution of P_syn
            fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
            ax1.hist(old_weights, density=True, color='b', label='data')
            ax1.plot(w_space, stats.norm.pdf(w_space, mean_old_weights, sigma_old_weights), label='curve fit', c='r', ls='--')
            ax1.set_xlabel('w')
            ax1.set_ylabel('density')
            ax1.legend()

            ax2.hist(P_syn, density=True, color='b')
            ax2.set_xlabel('P_syn')
            ax2.set_ylabel('density')

            # fit exponential curve over old_weights->P_syn scatter
            lowest = 0.0001
            ex_fu = lambda x, a, c, d: np.max(np.array([np.ones(shape=x.shape)*lowest, a*np.exp(c*x)+d]), axis=0)
            popt, pcov = curve_fit(ex_fu, old_weights, P_syn)
            print(", ".join([f'{p}={round(pv, 3)}' for (p, pv) in zip(['a', 'c', 'd'], popt)]))

            ax3.grid()
            ax3.scatter(old_weights, P_syn, alpha=0.3, label='data', c='b')
            ax3.plot(w_space, ex_fu(w_space, *popt), label='curve fit', c='r', ls='--')
            ax3.set_ylabel('P_syn')
            ax3.set_xlabel('w')
            ax3.legend()

            fig.suptitle('Weight Statistics of Original Network')
            plt.show()

            plt.scatter(P_syn, P_i, alpha=0.3)
            plt.show()

            # sample new weights and calculate corresponding traces
            new_source_rec = self.S_REC.source
            new_target_rec = self.S_REC.target

            # randomly sample new weights
            new_weights = np.random.normal(mean_old_weights, std_old_weights, size=(len(new_source_rec)))
            for i_syn, (so, ta) in enumerate(zip(new_source_rec, new_target_rec)):
                # get P_syn based on exponential relationship between w and P_syn
                current_w = new_weights[i_syn]
                new_P_syn = ex_fu(current_w, *popt)
                # calculate Pi and Pj from P_syn (hypothesis: they are equal)
                new_P_i = np.sqrt(new_P_syn/(10**current_w))
                assert(new_P_syn > 0. and new_P_i > 0.)
                # print(f'{current_w}, {new_P_syn}, {new_P_i}')
                # TODO set values for all traces in REC and S_REC
                self.S_REC.P_syn[i_syn] = new_P_syn
                self.S_REC.E_syn[i_syn] = new_P_syn
                self.S_REC.P_i[i_syn] = new_P_i
                self.S_REC.E_i[i_syn] = new_P_i
                self.S_REC.Z_i[i_syn] = new_P_i

                self.REC.P_j[ta] = new_P_i
                self.REC.E_j[ta] = new_P_i
                self.REC.Z_j[ta] = new_P_i

            self.REC.V_m[:] = np.random.uniform(-80, -60, size=(self.N)) * mV


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
        print(f'Total simulation time = {round(current_time, 2)} seconds.')

    def init_traces(self):
        eps = self.namespace['eps']
        print(f'Initialising model traces with eps={eps}')
        self.REC.set_states({'Z_j': eps, 'E_j': eps, 'P_j': eps})
        self.S_REC.set_states({'Z_i': eps, 'E_i': eps, 'P_i': eps, 'E_syn': eps**2, 'P_syn': eps**2})

    def save_traces(self, path):
        data = dict()

        # REC Layer
        # V_m
        data['V_m'] = self.REC.V_m/mV
        # I_w TODO compare if this is necessary
        data['I_w'] = self.REC.I_w/nA
        # Z_j
        data['Z_j'] = self.REC.Z_j/1
        # E_j
        data['E_j'] = self.REC.E_j/1
        # P_j
        data['P_j'] = self.REC.P_j/1

        # S_REC Synapses
        # connection
        data['S_source'] = np.array(self.S_REC.i)
        data['S_target'] = np.array(self.S_REC.j)
        # Z_i
        data['Z_i'] = self.S_REC.Z_i/1
        # E_i
        data['E_i'] = self.S_REC.E_i/1
        # P_i
        data['P_i'] = self.S_REC.P_i/1
        # E_syn
        data['E_syn'] = self.S_REC.E_syn/1
        # P_syn
        data['P_syn'] = self.S_REC.P_syn/1

        with open(path, 'wb') as f:
            pickle.dump(data, f)
            print('Successfully saved model data.')

    def check_inits(self):
        if self.namespace['stim_ta'] is None:
            raise ValueError('No stimulation TimedArray provided.')
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
        noise_pos_input = PoissonInput(target=self.REC, target_var='b_pos_noise', N=1, rate=self.namespace['r_bg'], weight=1)
        self.add_poisson(noise_pos_input, 'pos_noise')
        noise_neg_input = PoissonInput(target=self.REC, target_var='b_neg_noise', N=1, rate=self.namespace['r_bg'], weight=1)
        self.add_poisson(noise_neg_input, 'neg_noise')
        stim_input = PoissonInput(target=self.REC, target_var='g_stim', N=self.N_poisson, rate=self.namespace['r_stim'], weight=self.namespace['gr_stim'])
        self.add_poisson(stim_input, 'stim_input')

    def turn_on_imperfect(self, pattern:Pattern, noise_percentage=0.1, turn_on_other=False):
        b_on_array = np.zeros(shape=(self.N,))
        current_pyr = 0
        for hc in range(self.N_H):
            for mc in range(self.N_M):
                for i_pyr in range(current_pyr, current_pyr+self.N_pyr):
                    rs = np.random.random()
                    if pattern.contains(ColumnCoords(hc, mc)):
                        b_on_array[i_pyr] = (rs > (noise_percentage/2))
                    elif turn_on_other:
                        b_on_array[i_pyr] = (rs < (noise_percentage/2))
                current_pyr += self.N_pyr
        self.REC.b_on[:] = b_on_array
        return b_on_array

class TullyNetwork(CorticalNetwork):

    def __init__(self, namespace=tully_namespace, eqs=tully_equations):
        super().__init__(2, 1, 1, 0, namespace, eqs)

    # TODO overwrite all functions lol

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

                    
class MNGNetwork(CorticalNetwork):

    # N_pyr_total: theoretical number of total PYR neurons, 
    # although less will be simulated 
    # => N_pyr passed up to super constructor = N_pyr_total / N_MN
    # N_MN: how many neurons are represented by a single simulated neuron inside REC group
    def __init__(
            self, N_H, N_M, N_pyr_total, N_MN, N_BA, N_poisson=1,
            namespace=chr_namespace, eqs=chr_equations, filepath=None 
    ):
        
        super().__init__(N_H, N_M, int(N_pyr_total/N_MN), N_BA, namespace, eqs, filepath)