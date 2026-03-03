from brian2 import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from chr_params import chr_namespace
import sys
sys.path.append("./")
from brian_bcpnn.networks import ChrysanthidisNetwork
from brian_bcpnn.plot import trains, synapses, traces

prefs.codegen.target = 'numpy'
prefs.codegen.loop_invariant_optimisations = False
# np.seterr(all='raise')

defaultclock.dt = chr_namespace['t_sim']

N_hyper = 1
N_mini = 2
model = ChrysanthidisNetwork(N_hyper, N_mini)

# MONITORS
spikemon = SpikeMonitor(model.REC)
rec_statemon = StateMonitor(
    model.REC, ['V_m', 'Z_j', 'E_j', 'P_j', 'I_w', 'g_AMPA', 'I_AMPA', 'g_NMDA', 'I_NMDA', 'g_GABA', 'I_GABA', 'I_stim', 'beta', 'I_beta', 'g_stim'],
    record=True
    )
for mon in [spikemon, rec_statemon]:
    model.add_monitor(mon, mon.name)

bcpnn_synmon = StateMonitor(
    model.S_REC, ['Z_i', 'E_i', 'P_i', 'E_syn', 'P_syn', 'w', 'b_glut', 'clip_p_ratio'],
    record=True
)
model.add_monitor(bcpnn_synmon, bcpnn_synmon.name)

# model.activate_pattern([[0, 0]])
t_total = 1000*ms
model.run(t_total, namespace=chr_namespace)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True,
                                     gridspec_kw={'height_ratios': (1, 3, 3, 3)})

trains.compare_two_trains(ax1, spikemon, 0, 1)

traces.plot_z_traces(ax2, rec_statemon, bcpnn_synmon, model.S_REC, 0, 1)

traces.plot_e_traces(ax3, rec_statemon, bcpnn_synmon, model.S_REC, 0, 1)

traces.plot_p_traces(ax4, rec_statemon, bcpnn_synmon, model.S_REC, 0, 1)

ax4.set_xticks(np.arange(0, t_total / ms, 100))
ax4.set_xlabel("Time (ms)")
plt.show()

# plt.plot(rec_statemon.t/ms, rec_statemon.g_stim[0]/nS, label='g_stim')
# plt.plot(rec_statemon.t/ms, rec_statemon.V_m[0]/mV, label='n0 voltage')
# plt.plot(rec_statemon.t/ms, rec_statemon.V_m[1]/mV, label='n1 voltage')
plt.plot(rec_statemon.t/ms, rec_statemon.beta[0], label='n0 beta')
plt.plot(rec_statemon.t/ms, rec_statemon.beta[1], label='n1 beta')
# plt.plot(rec_statemon.t/ms, rec_statemon.I_AMPA[1]/nA, label='I_AMPA')
# plt.plot(rec_statemon.t/ms, rec_statemon.I_NMDA[1]/nA, label='I_NMDA')
# plt.plot(rec_statemon.t/ms, rec_statemon.I_GABA[1]/nA, label='I_GABA')
# plt.plot(rec_statemon.t/ms, rec_statemon.I_w[1]/nA, label='I_w')
# plt.plot(rec_statemon.t/ms, bcpnn_synmon[S_REC[0, 1]].w[0], label='w')
# plt.plot(rec_statemon.t/ms, bcpnn_synmon[S_REC[0, 1]].clip_p_prod[0], label='cp_prod')
# plt.plot(rec_statemon.t/ms, bcpnn_synmon[S_REC[0, 1]].clip_p_ratio[0], label='cp_ratio')
# plt.plot(rec_statemon.t/ms, bcpnn_synmon[S_REC[0, 1]].P_syn[0], label='P syn')
plt.legend()
plt.show()

# t_sample = 150 * ms
# n_samples = 8
# n_batches = 4
# t_batch = n_samples * t_sample
# tfinal = n_batches * t_batch

# # delete rec statemon for simulation run
# network.remove(rec_statemon)
# del rec_statemon

# del bcpnn_synmon
# bcpnn_synmon = StateMonitor(S_REC, ['w'], record=True)
# network.add(bcpnn_synmon)

# for i_batch in range(n_batches):
#     for i_sample in tqdm(range(n_samples)):
#         # clear run of 100 * ms
#         network.b_on = 0
#         network.run(100 * ms, namespace=chr_namespace)

#         # stimulation of current pattern
#         REC.b_on = [1 if i % 8 == i_sample else 0 for i in range(N_total)]
#         network.run(t_sample, namespace=chr_namespace)

#     # flush synmon
#     data = bcpnn_synmon.get_states(['w'])['w'][-1]
#     with open('data/chr/learned_weights.data', 'wb') as f:
#         pickle.dump(data, f)
#     if i_batch < (n_batches - 1):
#         del bcpnn_synmon
#         del data
#         # re-initialize synmon
#         bcpnn_synmon = StateMonitor(S_REC, ['w'], record=True)
#         network.add(bcpnn_synmon)


# fig, ax = plt.subplots()
# trains.get_full_train(ax, spikemon, N_total, tfinal)

# plt.show()

# fig, ax_array = plt.subplots(2, 4)
# for i, ax in enumerate(np.ndarray.flatten(ax_array)):
#     current_t = int(min(i/8 * t_batch / defaultclock.dt, t_batch / defaultclock.dt - 1))
#     ax.set_title(f't={current_t*defaultclock.dt}')
#     im = synapses.plot_weights_at_t(ax, bcpnn_synmon, S_REC, N_total, current_t)
#     fig.colorbar(im, ax=ax)

# plt.show()