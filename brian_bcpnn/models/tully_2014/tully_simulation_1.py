from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("./")
from brian_bcpnn.networks import TullyNetwork
from brian_bcpnn.plot import composite
from brian_bcpnn.utils.stim_utils import StimProtocol, ColumnCoords, StimTime
import brian_bcpnn.utils.stim_utils as stils
import brian_bcpnn.utils.synapse_utils as syls

model = TullyNetwork()

namespace = model.namespace
defaultclock.dt = namespace['sim_dt']

t_stim = 100 * ms
t_total = 500 * ms

# STIMULI
neuron_1_coords = ColumnCoords(0, 0)
neuron_2_coords = ColumnCoords(1, 0)
figure_2_stims = [
    StimProtocol(neuron_1_coords, StimTime(0*ms, 100*ms)),
    StimProtocol(neuron_2_coords, StimTime(100*ms, 200*ms)),
    StimProtocol(neuron_1_coords, StimTime(300*ms, 400*ms)),
    StimProtocol(neuron_2_coords, StimTime(300*ms, 400*ms))
]

model.namespace['stim_ta'] = stils.stim_times_to_timed_array(figure_2_stims, t_total, model.N_H, model.N_M)

# MONITORS
spikemon = model.add_spikemon()
tracemon = model.add_statemon(variables=model.REC_TRACES, record=True)
syn_tracemon = model.add_synmon(variables=model.S_REC_TRACES+['w'], record=True)

# SIMULATION
model.run(t_total)

composite.plot_traces(0, 1, spikemon, tracemon, syn_tracemon, model.S_REC, t_div=ms)
plt.show()