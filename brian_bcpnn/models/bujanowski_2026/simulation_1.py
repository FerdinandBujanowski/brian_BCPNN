# Toy BCPNN model based on Tully LIF neurons

from brian2 import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

import sys
sys.path.append("./")

from brian_bcpnn.models.tully_2014.parameters import * # change this to my own param file asp
import brian_bcpnn.helper as hlp
from brian_bcpnn.plot import trains, traces, synapses

from toy_model_1 import RecurrentLIF

# NETWORK PARAMETERS
N_hyper = 8
N_mini = 8
# N_total = N_hyper * N_mini

defaultclock.dt = sim_dt

model = RecurrentLIF(N_hyper=N_hyper, N_mini=N_mini)

# VISUALIZE LAT INH SYNAPSES
fig, ax = plt.subplots()
im = synapses.plot_connectivity(ax, model.S_LAT, model.N_total)
fig.colorbar(im, ax=ax)
plt.show()

t_sample = 150 * ms
n_samples = 8
n_batches = 4
t_batch = n_samples * t_sample
tfinal = n_batches * t_batch

for i_batch in range(n_batches):
    for i_sample in tqdm(range(n_samples)):
        model.REC.b_on = [1 if i % 8 == i_sample else 0 for i in range(model.N_total)]
        model.net.run(t_sample)

    # flush synmon
    data = model.bcpnn_synmon.get_states(['w'])
    with open('data/last_batch.data', 'wb') as f:
        pickle.dump(data, f)
    if i_batch < (n_batches - 1):
        del model.bcpnn_synmon
        del data
        # re-initialize synmon
        model.bcpnn_synmon = hlp.get_BCPNN_weight_synmon(model.S_REC, record=True)
        model.net.add(model.bcpnn_synmon)


fig, ax = plt.subplots()
trains.get_full_train(ax, model.spikemon, model.N_total, tfinal)

plt.show()

fig, ax_array = plt.subplots(2, 4)
for i, ax in enumerate(np.ndarray.flatten(ax_array)):
    current_t = int(min(i/8 * t_batch / defaultclock.dt, t_batch / defaultclock.dt - 1))
    ax.set_title(f't={current_t*defaultclock.dt}')
    im = synapses.plot_weights_at_t(ax, model.bcpnn_synmon, model.S_REC, model.N_total, current_t)
    fig.colorbar(im, ax=ax)

plt.show()