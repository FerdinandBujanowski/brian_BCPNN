# SIMULATION PURPOSE: Initialize weights of Fast-Slow network as N(0, 1) dist,
# 

from brian2 import *
import brian2cuda
set_device("cuda_standalone")

sys.path.append("./")
from brian_bcpnn.networks import TwoSynTypeNetwork
from brian_bcpnn.plot import trains
import brian_bcpnn.utils.stim_utils as stils

N_H = 10
N_M = 2
N_BA = 4
N_PYR = 30
model = TwoSynTypeNetwork(N_H, N_M, N_PYR, N_BA)

namespace = model.namespace
defaultclock.dt = namespace['t_sim']

t_total = 5000 * ms
model.namespace['eps'] = defaultclock.dt/t_total
sample_filepath = f'./data/fast-slow/{N_H}_{N_M}_{N_PYR}_init.data'
model.init_traces(sample_filepath)
model.namespace['stim_ta'] = stils.stim_times_to_timed_array([], t_total, model.N_H, model.N_M)

# MONITORS
basmon = model.add_basmon()
spikemon = model.add_spikemon()
curmon = model.add_statemon(variables=['I_beta', 'I_noise', 'I_AMPA', 'I_NMDA', 'I_GABA'], record=0)
weightmon = model.add_synmon(variables=['w'], record=list(range(100)))

model.run(t_total)

model.save_traces(f'./data/fast-slow/{N_H}_{N_M}_{N_PYR}_init.data')

fig, [ax0, ax1, ax2] = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': (1, 2, 2)})
trains.get_full_train(ax0, basmon, model.N_BA_total, t_total=t_total, t_div=ms, c='b')
trains.get_full_train(ax1, spikemon, model.N, t_total=t_total, t_div=ms)

ax2.plot(curmon.t/ms, curmon.I_beta[0]/nA, label='I_beta')
ax2.plot(curmon.t/ms, curmon.I_noise[0]/nA, label='I_noise')
ax2.plot(curmon.t/ms, curmon.I_AMPA[0]/nA, label='I_AMPA')
ax2.plot(curmon.t/ms, curmon.I_NMDA[0]/nA, label='I_NMDA')
ax2.plot(curmon.t/ms, curmon.I_GABA[0]/nA, label='I_GABA')
ax2.set_ylabel('Current/nA')
ax2.set_xlabel('Time/ms')
ax2.legend()
fig.suptitle('Dynamics without stimulation.')
plt.show()

# TODO same spike trains with average weight trajectory over time
fig, [ax0, ax1, ax2] = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': (1, 2, 2)})
trains.get_full_train(ax0, basmon, model.N_BA_total, t_total=t_total, t_div=ms, c='b')
trains.get_full_train(ax1, spikemon, model.N, t_total=t_total, t_div=ms)

weight_mean = np.mean(weightmon.w, axis=0)
weight_std = np.std(weightmon.w, axis=0)
n_std = 1
ax2.plot(weightmon.t/ms, weight_mean, c='b')
ax2.fill_between(weightmon.t/ms, weight_mean+n_std*weight_std, weight_mean-n_std*weight_std, color='b', alpha=0.3)
ax2.set_ylabel('Weight')
ax2.set_xlabel('Time/ms')
ax2.grid()
plt.show()


fig, ax = plt.subplots()
spikemon_freqs = trains.get_spiking_histogram(ax, spikemon, N=model.N, t_start=t_total/2, t_stop=t_total)
print(np.mean(spikemon_freqs))
print(np.std(spikemon_freqs))
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Count')
plt.title('PYR neuron spiking frequencies.')
plt.show()