from brian2 import *
import pickle
import time as tm
sys.path.append("./")
from brian_bcpnn.models.chrysanthidis_2025.chr_params import chr_namespace, chr_equations
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
            self.N, model=eqs['eqs_rec'], method='euler', threshold='V_m>V_peak', reset=eqs['reset_rec'], refractory='tau_ref'
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
        rec_trace_values = self.REC.get_states(self.REC_TRACES)
        for key in rec_trace_values.keys():
            if np.any(rec_trace_values[key] == 0.):
                raise ValueError('All trace variables should be initialized to above zero.')
        s_rec_trace_values = self.S_REC.get_states(self.S_REC_TRACES)
        for key in s_rec_trace_values.keys():
            if np.any(s_rec_trace_values[key] == 0.):
                raise ValueError('All trace variables should be initialized to above zero.')

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

