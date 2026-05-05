# SIMULATION PURPOSE: Load previously learnt weights and run idle / cue tests

from brian2 import *
sys.path.append("./")
from brian_bcpnn.networks import TwoSynTypeNetwork
from brian_bcpnn.models.bujanowski_2026.fiebig_params import fiebig_equations, fiebig_namespace
from brian_bcpnn.plot import trains, synapses, composite
from brian_bcpnn.stim_protocols.train_protocol import cue_n_epochs, get_total_time
import brian_bcpnn.utils.stim_utils as stils
import brian_bcpnn.utils.synapse_utils as syls

N_H = 6
N_M = 6
N_pyr = 30
N_BA = 4
N_batches = 1

model = TwoSynTypeNetwork(N_H, N_M, N_pyr=N_pyr, N_BA=N_BA, namespace=fiebig_namespace, eqs=fiebig_equations, filepath=f'./data/fast-slow/trained_6_6_30_copy.data')

model.namespace['kappa'] = 0

basmon = model.add_basmon()
spikemon = model.add_spikemon()
curmon = model.add_statemon(variables=['I_beta', 'I_noise', 'I_AMPA', 'I_NMDA', 'I_GABA', 'I_w', 'I_stim'], record=list(range(30)))
curmon_2 = model.add_statemon(variables=['I_beta', 'I_noise', 'I_AMPA', 'I_NMDA', 'I_GABA', 'I_w', 'I_stim'], record=list(range(30, 60)))

defaultclock.dt = model.namespace['t_sim']
t_start = 100 * ms
t_isi = 100 * ms
t_stim = 50 * ms
t_end = 100 * ms
N_batches = 1

pattern_list = stils.get_orthogonal_patterns(model.N_H, model.N_M)
pattern_list = stils.PatternList(pattern_list.patterns[0:2])
# pattern_list = stils.get_incomplete_patterns(pattern_list, 1)

t_total = get_total_time(t_start, t_stim, t_isi, t_end, N_batches, len(pattern_list.patterns))
model.namespace['tau_p'] = t_total
# model.init_traces(model='paper')

stims, t_total = cue_n_epochs(
    model, t_start, t_stim, t_isi, t_end,
    pattern_list, n_batches=N_batches
)
pt_dict = stils.get_pattern_time_dict(pattern_list, stims)

fig, [ax0, ax1, ax2, ax3] = plt.subplots(4, 1, sharex=True, gridspec_kw={'height_ratios': (1, 2, 3, 3)})
composite.plot_ba_pyr_trains(ax0, ax1, model, basmon, spikemon, t_total=t_total, pt_dict=pt_dict, t_div=second)

# average currents for first minicolumn
ax2.plot(curmon.t/second, np.mean(curmon.I_GABA[:]/nS, axis=0), label='GABA', color='r')
ax2.plot(curmon.t/second, np.mean(curmon.I_AMPA[:]/nS, axis=0), label='AMPA', color='g')
ax2.plot(curmon.t/second, np.mean(curmon.I_NMDA[:]/nS, axis=0), label='NMDA', color='g', ls=':')
ax2.plot(curmon.t/second, np.mean(curmon.I_beta[:]/nS, axis=0), label='beta', color='b')
ax2.plot(curmon.t/second, np.mean(curmon.I_noise[:]/nS, axis=0), label='noise', color='k', ls='--')
ax2.plot(curmon.t/second, np.mean(curmon.I_stim[:]/nS, axis=0), label='stim', color='k', ls=':')
ax2.plot(curmon.t/second, np.mean(curmon.I_w[:]/nS, axis=0), label='adapt', color='b', ls='--')
ax2.legend()
ax3.set_xlabel(f'Time/{second}')

# average currents for second minicolumn
ax3.plot(curmon_2.t/second, np.mean(curmon_2.I_GABA[:]/nS, axis=0), label='GABA', color='r')
ax3.plot(curmon_2.t/second, np.mean(curmon_2.I_AMPA[:]/nS, axis=0), label='AMPA', color='g')
ax3.plot(curmon_2.t/second, np.mean(curmon_2.I_NMDA[:]/nS, axis=0), label='NMDA', color='g', ls=':')
ax3.plot(curmon_2.t/second, np.mean(curmon_2.I_beta[:]/nS, axis=0), label='beta', color='b')
ax3.plot(curmon_2.t/second, np.mean(curmon_2.I_noise[:]/nS, axis=0), label='noise', color='k', ls='--')
ax3.plot(curmon_2.t/second, np.mean(curmon_2.I_stim[:]/nS, axis=0), label='stim', color='k', ls=':')
ax3.plot(curmon_2.t/second, np.mean(curmon_2.I_w[:]/nS, axis=0), label='adapt', color='b', ls='--')
ax3.legend()
ax3.set_xlabel(f'Time/{second}')
plt.show()

# plot firing rate averages over time
fig, [ax0, ax1, ax2] = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': (1, 2, 3)})
composite.plot_ba_pyr_trains(ax0, ax1, model, basmon, spikemon, t_total=t_total, pt_dict=pt_dict, t_div=second)

pattern_1_train = trains.get_pattern_population_train(spikemon, model, pattern_list.patterns[0], t_total, defaultclock.dt)
pattern_1_freqs = trains.get_firing_rate_estimate(pattern_1_train, N_M*N_pyr, 0.1, t_total/ms, kernel_size=15)
ax2.plot(curmon_2.t/second, pattern_1_freqs, label='Pattern 1 Avg Freqs', c='g')

pattern_2_train = trains.get_pattern_population_train(spikemon, model, pattern_list.patterns[1], t_total, defaultclock.dt)
pattern_2_freqs = trains.get_firing_rate_estimate(pattern_2_train, N_M*N_pyr, 0.1, t_total/ms, kernel_size=15)
ax2.plot(curmon_2.t/second, pattern_2_freqs, label='Pattern 2 Avg Freqs', c='r')

ax2.legend()
ax2.set_xlabel('Time/second')
ax2.set_ylabel('Firing Rate (Hz)')

plt.show()

# test on first MC
# mc_0_train = trains.get_minicolumn_population_train(spikemon, model, 0, 0, t_total, defaultclock.dt)
# mc_0_freqs = trains.get_firing_rate_estimate(mc_0_train, N_pyr, 0.1, t_total/ms, kernel_size=15)
# plt.plot(curmon_2.t/second, mc_0_freqs)
# plt.show()

# weight matrix
fig, ax = plt.subplots()
im = synapses.plot_weights(ax, model.S_REC, model.N)
fig.colorbar(im, ax=ax)
plt.title('BCPNN weight matrix')
plt.show()

# minicolumn average weight matrix
fig, ax = plt.subplots()
im = synapses.plot_weight_matrix_averages(ax, model)
fig.colorbar(im, ax=ax)
plt.title('Average minicolumn weights')
plt.show()

fig, [ax1, ax2] = plt.subplots(1, 2)
trains.get_spiking_histogram(ax1, spikemon, model.N, t_start=t_start+t_stim, t_stop=t_start+2*t_stim)
ax1.set_title('PYR spikes during stim')
ax1.set_xlabel('Freq/Hz')
basmon_freqs = trains.get_spiking_histogram(ax2, basmon, model.N_BA_total, t_start=t_start, t_stop=t_start+t_stim)
ax2.set_xlabel('Freq/Hz')
ax2.set_title('BA spikes during stim')
ax1.set_ylabel('Count')
plt.show()