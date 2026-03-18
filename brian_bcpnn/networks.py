from brian2 import *
import pickle
sys.path.append("./")
from brian_bcpnn.models.chrysanthidis_2025.chr_params import chr_namespace
import brian_bcpnn.models.chrysanthidis_2025.chr_model as eqs
import brian_bcpnn.utils.synapse_utils as syls

class CorticalNetwork():

    def __init__(self, N_hyper, N_mini, N_pyr=1, N_basket=0, namespace=None):

        self.N_hyper = N_hyper
        self.N_mini = N_mini
        self.N_pyr = N_pyr
        self.N = N_hyper * N_mini * N_pyr

        self.N_basket = N_basket
        self.N_basket_total = N_hyper * N_mini * N_basket
        self.monitors = dict()
        self.inputs = dict()
        self.network = Network()
        if namespace is not None:
            self.set_namespace(namespace)
            self.namespace['N_pyr'] = N_pyr

        self.REC = None
        self.S_REC = None
        self.BA = None
        self.S_PB = None
        self.S_BP = None

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

    def delete_poisson(self, name):
        poisson_input = self.inputs.pop(name)
        self.network.remove(poisson_input)
        del poisson_input

    def set_namespace(self, namespace):
        self.namespace = namespace

    def run(self, time, namespace=None):
        if namespace is not None:
            self.network.run(time, namespace=namespace)
        elif self.namespace is not None:
            self.network.run(time, namespace=self.namespace)
        else:
            self.network.run(time)

class ChrysanthidisNetwork(CorticalNetwork):

    def __init__(self, N_hyper, N_mini, N_pyr, N_basket=0, namespace=chr_namespace, N_poisson=1, filepath=None):
        super().__init__(N_hyper, N_mini, N_pyr, N_basket, namespace)
        self.namespace['N_pyr'] = N_pyr
    
        # RECURRENT HYPER-MINI-COLUMN LAYER
        self.REC = NeuronGroup(
            self.N, model=eqs.eqs_rec, method='euler', threshold='V_m>V_t', reset=eqs.reset_rec, refractory='tau_ref'
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
                self.data = data
        else:
            self.REC.V_m = self.namespace['E_L']
            # self.REC.P_j = self.namespace['eps']

        # RECURRENT LAYER POISSON INPUT
        noise_pos_input = PoissonInput(target=self.REC, target_var='g_bg', N=1, rate=self.namespace['r_bg'], weight=self.namespace['gr_bg'])
        self.add_poisson(noise_pos_input, 'pos_noise')
        noise_neg_input = PoissonInput(target=self.REC, target_var='g_bg', N=1, rate=self.namespace['r_bg'], weight=self.namespace['gr_bg_n'])
        self.add_poisson(noise_neg_input, 'neg_noise')
        stim_input = PoissonInput(target=self.REC, target_var='g_stim', N=N_poisson, rate=self.namespace['r_stim'], weight=self.namespace['gr_stim'])
        self.add_poisson(stim_input, 'stim_input')

        # RECURRENT LAYER BCPNN SYNAPSES
        # TODO sample i-j distances + synaptic delay
        self.S_REC = Synapses(
            self.REC, self.REC, model=eqs.bcpnn_syn_model, on_pre=eqs.bcpnn_syn_on_pre, method='euler', delay=self.namespace['t_delay']
        )
        self.network.add(self.S_REC)

        # create connections
        if hasattr(self, 'data'):
            source_rec = self.data['S_source']
            target_rec = self.data['S_target']
            self.S_REC.connect(i=source_rec, j=target_rec)

            self.S_REC.Z_i = self.data['Z_i']
            self.S_REC.E_i = self.data['E_i']
            self.S_REC.P_i = self.data['P_i']
            self.S_REC.E_syn = self.data['E_syn']
            self.S_REC.P_syn = self.data['P_syn']
        else:
            source_rec, target_rec = syls.get_rec_synapses(
                self.N_hyper, self.N_mini, self.N_pyr, 
                # 0.9, 0.5, 0.1 # to test connectivity
                cp_same_mini=self.namespace['cp_PP'],
                cp_same_hyper=self.namespace['cp_PPL'],
                cp_diff_hyper=self.namespace['cp_PPL']
            )
            self.S_REC.connect(i=source_rec, j=target_rec)
            # self.S_REC.P_i = self.namespace['eps']

        # BASKET CELLS
        self.BA = NeuronGroup(
            self.N_basket_total, model=eqs.eqs_basket, method='euler', threshold='V_m>V_t', reset=eqs.reset_ba, refractory='tau_ref'
        )
        self.network.add(self.BA)
        self.BA.V_m = self.namespace['E_L']

        # BASKET CELL SYNAPSES
        (sP, tB, sB, tP) = syls.get_basket_synapses(
            self.N_hyper, self.N_mini, self.N_pyr, self.N_basket,
            cp_PB=self.namespace['cp_PB'],
            cp_BP=self.namespace['cp_BP'],
            symmetry=True # TODO ask if this is correct / necessary
        )
        # PYR --> BA
        self.S_PB = Synapses(
            self.REC, self.BA, on_pre=eqs.pyr_basket_on_pre, method='euler', delay=self.namespace['t_delay']
        )
        # BA --> PYR
        self.S_BP = Synapses(
            self.BA, self.REC, on_pre=eqs.basket_pyr_on_pre, method='euler', delay=self.namespace['t_delay']
        )
        self.S_BP.connect(i=sB, j=tP)
        self.S_PB.connect(i=sP, j=tB)
        self.network.add([self.S_PB, self.S_BP])

        # fig, [ax1, ax2] = plt.subplots(1, 2)
        # synapses.plot_connectivity(ax1, self.S_PB, self.N, self.N_basket_total)
        # ax1.set_title('PYR-BA connectivity')
        # synapses.plot_connectivity(ax2, self.S_BP, self.N_basket_total, self.N)
        # ax2.set_title('BA-PYR connectivity')
        # plt.show()

        # MONITORS
        # ... to be added in file instantiating class

    def run(self, time):
        if self.namespace['stim_ta'] is None:
            raise ValueError('No stimulation TimedArray provided.')
        else:
            super().run(time, self.namespace)

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