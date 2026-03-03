from brian2 import *
sys.path.append("./")
from brian_bcpnn.models.chrysanthidis_2025.chr_params import chr_namespace
import brian_bcpnn.models.chrysanthidis_2025.chr_model as eqs

class CorticalNetwork():

    def __init__(self, N_hyper, N_mini, N_pyr=1, N_basket=0, namespace=None):

        self.N_hyper = N_hyper
        self.N_mini = N_mini
        self.N_pyr = N_pyr
        self.N = N_hyper * N_mini * N_pyr

        self.N_basket = N_basket
        self.monitors = dict()
        self.inputs = dict()
        self.network = Network()
        if namespace is not None:
            self.set_namespace(namespace)

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

    def __init__(self, N_hyper, N_mini, N_basket=0, namespace=chr_namespace):
        super().__init__(N_hyper, N_mini, N_basket, namespace)
    
        # RECURRENT HYPER-MINI-COLUMN LAYER
        self.REC = NeuronGroup(
            self.N, model=eqs.eqs_rec, method='euler', threshold='V_m>V_t', reset=eqs.reset_rec, refractory='tau_ref'
        )
        self.network.add(self.REC)
        self.REC.V_m = self.namespace['E_L']
        self.P_j = self.namespace['eps']

        # BCPNN SYNAPSES
        # TODO sample i-j distances + synaptic delay
        self.S_REC = Synapses(
            self.REC, self.REC, model=eqs.bcpnn_syn_model, on_pre=eqs.bcpnn_syn_on_pre, method='euler', delay=1*ms
        )
        self.network.add(self.S_REC)
        self.S_REC.connect(condition='i!=j')
        self.S_REC.P_i = self.namespace['eps']
        self.S_REC.P_syn = self.namespace['eps']

        # S_INH.connect(i=source_inh, j=target_inh)

        # POISSON INPUT
        noise_pos_input = PoissonInput(target=self.REC, target_var='g_bg', N=1, rate=self.namespace['r_bg'], weight=self.namespace['gr_bg'])
        self.add_poisson(noise_pos_input, 'pos_noise')
        noise_neg_input = PoissonInput(target=self.REC, target_var='g_bg', N=1, rate=self.namespace['r_bg'], weight=self.namespace['gr_bg_n'])
        self.add_poisson(noise_neg_input, 'neg_noise')
        stim_input = PoissonInput(target=self.REC, target_var='g_stim', N=5, rate=self.namespace['r_stim'], weight=self.namespace['gr_stim'])
        self.add_poisson(stim_input, 'stim_input')

        # MONITORS
        # ... to be added in file instantiating class

    def turn_off_all(self):
        self.REC.b_on[:] = 0
    
    def activate_pattern(self, pattern, turn_off_before=True):
        if turn_off_before:
            self.turn_off_all()
        for [hy, mi] in pattern:
            self.REC.b_on[hy*int(self.N/self.N_hyper)+mi] = 1 

class EquationSystem():
    def __init__(self, equation_string):
        self.eqs = equation_string

    def __str__(self):
        return self.eqs

    def add(self, equation_string):
        self.eqs = f'''{self.eqs}
        {equation_string}'''
        return self
    
