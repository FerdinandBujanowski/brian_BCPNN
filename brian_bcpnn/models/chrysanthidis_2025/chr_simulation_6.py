# SIMULATION PURPOSE: Load previously saved network data of a 4x2x30 network,
# Run tests on non stimulation and stimulation dynamics

from brian2 import *
sys.path.append("./")
from brian_bcpnn.networks import ChrysanthidisNetwork
from brian_bcpnn.models.chrysanthidis_2025.chr_params import chr_namespace
from brian_bcpnn.plot import trains, synapses
from brian_bcpnn.utils.stim_utils import add_time, StimProtocol, ColumnCoords
import brian_bcpnn.utils.stim_utils as stils

namespace = chr_namespace
defaultclock.dt = namespace['t_sim']

N_hyper = 4
N_mini = 2
N_pyr = 30
N_basket = 4
t_stim, T_stim = [chr_namespace[s] for s in ['t_stim', 'T_stim']]

model = ChrysanthidisNetwork(
    N_hyper, N_mini, N_pyr=N_pyr, N_basket=N_basket, 
    namespace=namespace, filepath='data/chr/stable_init_test.data', N_poisson=2
)

spikemon = SpikeMonitor(model.REC)
model.add_monitor(spikemon, 'spikemon')
basmon = SpikeMonitor(model.BA)
model.add_monitor(basmon, 'basmon')

# t_total = 0*ms
# Plot connectivity and initial weights distribution from file
# t_total, t_init = add_time(t_total, 1*ms)
# model.run(t_init) # run a tiny bit otherwise all weights are = 0
# fig, [ax1, ax2] = plt.subplots(1, 2)
# synapses.hist_presyn_count(ax1, model.S_REC, model.N)
# im = synapses.plot_weights(ax2, model.S_REC, model.N)
# fig.colorbar(im, ax=ax2)
# plt.show()

# get orthogonal patterns
orth_patterns = stils.get_orthogonal_patterns(N_hyper, N_mini)
first_pattern = orth_patterns.subset(0)
t_init = 1*second
t_stim = 1*second
t_end = .5*second
stims, t_total = stils.train_patterns_protocol(first_pattern, t_init, t_stim, 0*ms, t_end)
model.namespace['stim_ta'] = stils.stim_times_to_timed_array(stims, t_total, N_hyper, N_mini)
model.run(t_total)

# TODO turn plot into 3 part plot (raster, hist before stim, hist after stim)
fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
trains.get_full_train(ax1, spikemon, model.N, t_total, t_div=second)
ax1.set_xlabel('Time (seconds)')

freqs = trains.get_spiking_histogram(ax2, spikemon, model.N, t_start=0*ms, t_stop=t_init)
# print(round(np.mean(freqs), 2))
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Neuron Count')
ax2.set_title('Firing Frequency Distribution Without Stimulus')

freqs = trains.get_spiking_histogram(ax3, spikemon, model.N, t_start=t_init, t_stop=t_init+t_stim)
# print(round(np.mean(freqs), 2))
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Neuron Count')
ax3.set_title('Firing Frequency Distribution With Stimulus')
plt.show()

ax1 = plt.subplot2grid((3, 6), (0, 0), colspan=2, rowspan=1)
ax2 = plt.subplot2grid((3, 6), (1, 0), colspan=2, rowspan=2, sharex=ax1)
ax3 = plt.subplot2grid((3, 6), (0, 2), colspan=2, rowspan=3)
ax4 = plt.subplot2grid((3, 6), (0, 4), colspan=2, rowspan=3, sharey=ax3)

trains.get_full_train(ax1, basmon, model.N_basket_total, t_total, t_div=second, c='b')
ax1.set_ylabel('# BA neuron')
trains.get_full_train(ax2, spikemon, model.N, t_total, t_div=second)
ax2.set_ylabel('# PYR neuron')
ax2.set_xlabel('Time (second)')

freqs = trains.get_spiking_histogram(ax3, basmon, model.N_basket_total, t_start=0*ms, t_stop=t_init)
# print(round(np.mean(freqs), 2))
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Neuron Count')
ax3.set_title('BA Spiking Dist (No Stimulus)')

freqs = trains.get_spiking_histogram(ax4, basmon, model.N_basket_total, t_start=t_init, t_stop=t_init+t_stim)
# print(round(np.mean(freqs), 2))
ax4.set_xlabel('Frequency (Hz)')
ax4.set_title('BA Spiking Dist (Stimulus)')

plt.tight_layout()
plt.show()