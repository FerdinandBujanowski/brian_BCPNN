# SIMULATION PURPOSE: Load previously learnt weights and run idle / cue tests

from brian2 import *
sys.path.append("./")
from brian_bcpnn.networks import TwoSynTypeNetwork
from brian_bcpnn.models.chrysanthidis_2025.fiebig_params import fiebig_equations, fiebig_namespace
from brian_bcpnn.plot import trains, synapses, composite
from brian_bcpnn.stim_protocols.train_protocol import train_n_epochs, get_total_time
import brian_bcpnn.utils.stim_utils as stils
import brian_bcpnn.utils.synapse_utils as syls

# set_device('cpp_standalone')

N_H = 6
N_M = 6
N_pyr = 30
N_BA = 4
N_batches = 1

model = TwoSynTypeNetwork(N_H, N_M, N_pyr=N_pyr, N_BA=N_BA, namespace=fiebig_namespace, eqs=fiebig_equations)#, filepath=f'./data/fast-slow/trained_{N_H}_{N_M}_{N_pyr} copy.data')


model.namespace['K_AMPA'] = 0
model.namespace['K_NMDA'] = 0

basmon = model.add_basmon()
spikemon = model.add_spikemon()

defaultclock.dt = model.namespace['t_sim']
t_start = 100 * ms
t_isi = 200 * ms
t_stim = 50 * ms
t_end = 100 * ms
N_batches = 1

pattern_list = stils.get_orthogonal_patterns(model.N_H, model.N_M)
pattern_list = stils.PatternList(pattern_list.patterns[0:2])
# pattern_list = stils.get_incomplete_patterns(pattern_list, 1)

t_total = get_total_time(t_start, t_stim, t_isi, t_end, N_batches, len(pattern_list.patterns))
# model.namespace['eps'] = defaultclock.dt/t_total
model.init_traces(model='paper')

stims, t_total = train_n_epochs(
    model, t_start, t_stim, t_isi, t_end,
    pattern_list, n_batches=N_batches
)
pt_dict = stils.get_pattern_time_dict(pattern_list, stims)

fig, [ax0, ax1] = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': (1, 2)})
composite.plot_ba_pyr_trains(ax0, ax1, model, basmon, spikemon, t_total=t_total, pt_dict=pt_dict, t_div=ms)
ax1.set_xlabel(f'Time/{ms}')
plt.show()

fig, [ax1, ax2] = plt.subplots(1, 2)
trains.get_spiking_histogram(ax1, spikemon, model.N, t_start=t_start, t_stop=t_start+t_stim)
ax1.set_title('PYR spikes during stim')
ax1.set_xlabel('Freq/Hz')
basmon_freqs = trains.get_spiking_histogram(ax2, basmon, model.N_BA_total, t_start=t_start, t_stop=t_start+t_stim)
ax2.set_xlabel('Freq/Hz')
ax2.set_title('BA spikes during stim')
ax1.set_ylabel('Count')
plt.show()

# fig, [ax1, ax2] = plt.subplots(1, 2)
# im1 = synapses.plot_weights(ax1, model.S_REC, model.N)
# ax1.set_title('AMPA weights')
# im2 = synapses.plot_weights(ax2, model.S_NMDA, model.N)
# ax2.set_title('NMDA weights')
# fig.colorbar(im1, ax=ax1)
# fig.colorbar(im2, ax=ax2)
# plt.show()