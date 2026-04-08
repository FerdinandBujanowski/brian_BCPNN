from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("./")
from brian_bcpnn.networks import TullyNetwork
from brian_bcpnn.plot import composite
from brian_bcpnn.utils.stim_utils import StimProtocol, ColumnCoords, StimTime
import brian_bcpnn.utils.stim_utils as stils
import brian_bcpnn.utils.synapse_utils as syls

model = TullyNetwork() #Everything defined. Subclass of CorticalNetwork.

namespace = model.namespace # using defined attribute namespace from the TullyNetwork. 
# namespace used in Brian as dictionary of defined parameters and values. 
defaultclock.dt = namespace['sim_dt'] #using namespace but specifically the value of sim_dt (sim_dt : 0.1 * ms (key-value pair)) in tully_params file. 

t_stim = 100 * ms #using "blocks" of 100ms for simulation.
t_total = 500 * ms #total simulation time.

# STIMULI
# ColumnCoords, StimProtocol, StimTime all taken from stim_utils file.
neuron_1_coords = ColumnCoords(0, 0) # first neuron, pre-synaptic.
neuron_2_coords = ColumnCoords(1, 0) # second neuron, post-synaptic.
figure_2_stims = [
    StimProtocol(neuron_1_coords, StimTime(t_start=0*ms, t_end=100*ms)),   #pre-syn neuron 0-100ms
    StimProtocol(neuron_2_coords, StimTime(t_start=100*ms, t_end=200*ms)), #post-syn neuron 100-200ms
    StimProtocol(neuron_1_coords, StimTime(t_start=300*ms, t_end=400*ms)), #none 200-300ms, then both 300-400ms
    StimProtocol(neuron_2_coords, StimTime(t_start=300*ms, t_end=400*ms))  
]

#actually overwriting - check what it is written as. -------
model.namespace['stim_ta'] = stils.stim_times_to_timed_array(figure_2_stims, t_total, model.N_H, model.N_M)

# MONITORS
spikemon = model.add_spikemon()
tracemon = model.add_statemon(variables=model.REC_TRACES, record=True)
syn_tracemon = model.add_synmon(variables=model.S_REC_TRACES+['w'], record=True)

# SIMULATION
model.run(t_total)

#0 is i, 1 is j 
composite.plot_traces(0, 1, spikemon, tracemon, syn_tracemon, model.S_REC, t_div=ms)
plt.show()