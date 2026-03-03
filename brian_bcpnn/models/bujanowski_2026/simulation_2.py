# Toy BCPNN model based on Tully LIF neurons
# Simulation 2: trying to load previously learnt weights into the model 
# and test its error correction capabilities

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

import sys
sys.path.append("./")

from brian_bcpnn.models.bujanowski_2026.parameters import SIM_PARAM_INIT
import brian_bcpnn.helper as hlp
from brian_bcpnn.plot import trains, traces, synapses

from toy_model_1 import RecurrentLIF

# NETWORK PARAMETERS
N_hyper = 8
N_mini = 8
defaultclock.dt = SIM_PARAM_INIT['sim_dt']

model = RecurrentLIF(N_hyper=N_hyper, N_mini=N_mini, complex_synmon=True)

w_saved = None
with open('data/last_batch.data', 'rb') as f:
    data = pickle.load(f)
    w_saved = data

model.S_REC.w = w_saved
# print(model.S_REC.w)

model.REC.b_on[0] = 1

t_total = 1000 * ms
model.net.run(t_total)

fig, ax = plt.subplots()
trains.get_full_train(ax, model.spikemon, model.N_total, t_total)
plt.show()

freqs = []
for n_neuron in range(N_hyper*N_mini):
    freq = trains.get_neuron_frequency(model.spikemon, n_neuron, t_stop=t_total) / Hz
    freqs.append(freq)
plt.hist(freqs)
plt.show()


plt.plot(model.rec_statemon.t/ms, model.rec_statemon.V[8]/mV)
# plt.plot(model.rec_statemon.t/ms, model.rec_statemon.g_ex[8]/nS)
# plt.plot(model.rec_statemon.t/ms, model.bcpnn_synmon[model.S_REC[0,8]].w[0], label='w')
plt.legend()
plt.show()

# PLOTTING I-F curve
# I_array = [.1, .2, .25, .2501, .2505, .251, .252, .253, .254, .255, .26, .27, .28, .29, .3, .5, 1]
# F_array = [0, 0, 0, 8, 12, 14, 16, 18, 20, 21, 27, 37, 45, 52, 59, 158, 285]
# plt.plot(I_array, F_array, marker='o', ls='--')
# plt.xlabel('Input current / -1 (nA)')
# plt.ylabel('Firing Frequency (Hz)')
# plt.show()