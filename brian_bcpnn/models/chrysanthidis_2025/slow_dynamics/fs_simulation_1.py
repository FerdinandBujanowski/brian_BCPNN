# SIMULATION PURPOSE: Initialize weights of Fast-Slow network as N(0, 1) dist,
# 

from brian2 import *
# import brian2cuda
# set_device("cuda_standalone")

sys.path.append("./")
from brian_bcpnn.networks import TwoSynTypeNetwork
from brian_bcpnn.plot import trains, synapses
from brian_bcpnn.models.chrysanthidis_2025.fiebig_params import fiebig_namespace, fiebig_equations
import brian_bcpnn.utils.stim_utils as stils

N_H = 6
N_M = 6
N_BA = 4
N_PYR = 30
model = TwoSynTypeNetwork(N_H, N_M, N_PYR, N_BA, namespace=fiebig_namespace, eqs=fiebig_equations)

namespace = model.namespace
defaultclock.dt = namespace['t_sim']

t_total = 0.5 * second
spike_freq = namespace['f_min']
model.namespace['eps'] = spike_freq / namespace['f_max']
# sample_filepath = f'./data/fast-slow/10_2_30_init.data'
model.init_traces(model='zero_weight', baseline=spike_freq)
model.namespace['stim_ta'] = stils.stim_times_to_timed_array([], t_total, model.N_H, model.N_M)

# TODO comment these out
model.namespace['K_AMPA'] = 0
model.namespace['K_NMDA'] = 0

# MONITORS
basmon = model.add_basmon()
spikemon = model.add_spikemon()
curmon = model.add_statemon(variables=['I_beta', 'I_noise', 'I_AMPA', 'I_NMDA', 'I_GABA', 'I_w', 'g_GABA', 'g_BA'], record=0)
weightmon = model.add_synmon(variables=['w'], record=list(range(100)))


bas_statemon = StateMonitor(model.BA, variables=['V_m', 'g_ex'], record=0)
model.add_monitor(bas_statemon, bas_statemon.name)

all_pre_pb = [i for (i,j) in zip(model.S_PB.i, model.S_PB.j) if j==0]
bas_synmon = StateMonitor(model.S_PB, variables=['H_ex'], record=all_pre_pb)
model.add_monitor(bas_synmon, bas_synmon.name)

voltage_statemon = StateMonitor(model.REC, variables=['V_m', 'g_BA'], record=0)
model.add_monitor(voltage_statemon, voltage_statemon.name)

# print([f'{i}->{j}' for i, j in zip(model.S_PB.i, model.S_PB.j) if j==0])

model.run(t_total)

# model.save_traces(f'./data/fast-slow/{N_H}_{N_M}_{N_PYR}_init.data')

# fig, [ax1, ax2] = plt.subplots(1, 2)
# im1 = synapses.plot_weights(ax1, model.S_REC, model.N)
# ax1.set_title('AMPA weights')
# im2 = synapses.plot_weights(ax2, model.S_NMDA, model.N)
# ax2.set_title('NMDA weights')
# fig.colorbar(im1, ax=ax1)
# fig.colorbar(im2, ax=ax2)
# plt.show()

fig, [ax0, ax1, ax2] = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': (1, 2, 2)})
trains.get_full_train(ax0, basmon, model.N_BA_total, t_total=t_total, t_div=ms, c='b')
trains.get_full_train(ax1, spikemon, model.N, t_total=t_total, t_div=ms)

ax2.plot(curmon.t/ms, curmon.I_beta[0]/nA, label='I_beta')
ax2.plot(curmon.t/ms, curmon.I_noise[0]/nA, label='I_noise')
ax2.plot(curmon.t/ms, curmon.I_AMPA[0]/nA, label='I_AMPA')
ax2.plot(curmon.t/ms, curmon.I_NMDA[0]/nA, label='I_NMDA')
ax2.plot(curmon.t/ms, curmon.I_w[0]/nA, label='I_w')
ax2.plot(curmon.t/ms, curmon.I_GABA[0]/nA, label='I_GABA')
ax2.set_ylabel('Current/nA')
ax2.set_xlabel('Time/ms')
ax2.legend()
fig.suptitle('Dynamics without stimulation.')
plt.show()

fig, [ax0, ax1, ax2] = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': (1, 2, 2)})
trains.get_full_train(ax0, basmon, model.N_BA_total, t_total=t_total, t_div=ms, c='b')
trains.get_full_train(ax1, spikemon, model.N, t_total=t_total, t_div=ms)

ax2.plot(curmon.t/ms, curmon.g_GABA[0]/nS, label='g_GABA')
ax2.plot(curmon.t/ms, curmon.g_BA[0]/nS, label='g_BA')
ax2.set_ylabel('Conductance/nS')
ax2.set_xlabel('Time/ms')
ax2.legend()
fig.suptitle('Dynamics without stimulation.')
plt.show()

# fig, [ax0, ax1, ax2] = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': (1, 2, 2)})
# trains.get_full_train(ax0, basmon, model.N_BA_total, t_total=t_total, t_div=ms, c='b')
# trains.get_full_train(ax1, spikemon, model.N, t_total=t_total, t_div=ms)

# weight_mean = np.mean(weightmon.w, axis=0)
# weight_std = np.std(weightmon.w, axis=0)
# n_std = 1
# ax2.plot(weightmon.t/ms, weight_mean, c='b')
# ax2.fill_between(weightmon.t/ms, weight_mean+n_std*weight_std, weight_mean-n_std*weight_std, color='b', alpha=0.3)
# ax2.set_ylabel('Weight')
# ax2.set_xlabel('Time/ms')
# ax2.grid()
# plt.show()


fig, ax = plt.subplots()
# TODO add regularity plot from angeliki to this graph
spikemon_freqs = trains.get_spiking_histogram(ax, spikemon, N=model.N, t_start=t_total/2, t_stop=t_total)
print(np.mean(spikemon_freqs))
print(np.std(spikemon_freqs))
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Count')
plt.title('PYR neuron spiking frequencies.')
plt.show()

fig, ax = plt.subplots()
plotted_basket_var = bas_statemon.V_m[0]/mV
# plotted_basket_var = bas_statemon.g_ex[0]/nS
ax.plot(bas_statemon.t/ms, plotted_basket_var, color='b')
spike_trains = spikemon.spike_trains()
all_presyn = all_pre_pb
for i in all_presyn:
    current_train = spike_trains[i]
    # ax.plot(bas_synmon.t/ms, bas_synmon[i].H_ex, color='g')
    ax.scatter(current_train/ms, [plotted_basket_var[int(t/defaultclock.dt)] for t in current_train], color='r')
plt.title('Basket cell membrane potential evolution.')
plt.xlabel('Time/ms')
plt.ylabel('Voltage/mV')
plt.show()

fig, ax = plt.subplots()
ax.plot(voltage_statemon.t/ms, voltage_statemon.V_m[0]/mV, color='k')
all_presyn = [i for i, j in zip(model.S_BP.i, model.S_BP.j) if j == 0]
spike_trains = basmon.spike_trains()
for i in all_presyn:
    current_train = spike_trains[i]
    ax.scatter(current_train/ms, [voltage_statemon.V_m[0][int(t/defaultclock.dt)]/mV for t in current_train], color='b')
plt.title('PYR cell membrane potential evolution.')
plt.xlabel('Time/ms')
plt.ylabel('Voltage/mV')
plt.show()