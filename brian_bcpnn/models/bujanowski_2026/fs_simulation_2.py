# SIMULATION PURPOSE: Load previously initialised model params (with same architecture)
# and train orthogonal patterns for several batches - save final parameters to file 

from brian2 import *
sys.path.append("./")
from brian_bcpnn.networks import TwoSynTypeNetwork
from brian_bcpnn.models.bujanowski_2026.fiebig_params import fiebig_namespace, fiebig_equations
from brian_bcpnn.plot import trains, synapses, composite
from brian_bcpnn.stim_protocols.train_protocol import cue_n_epochs, get_total_time
import brian_bcpnn.utils.stim_utils as stils
import brian_bcpnn.utils.synapse_utils as syls

N_H = 6
N_M = 6
N_pyr = 30
N_BA = 4
N_batches = 2

model = TwoSynTypeNetwork(N_H, N_M, N_pyr=N_pyr, N_BA=N_BA, namespace=fiebig_namespace, eqs=fiebig_equations)

# SYNAPSE INDEXING
mc_range_i = range(N_pyr) # home MC for comparison
mc_range_j1 = range(N_M*N_pyr,(N_M+1)*N_pyr) # same activation MC for comparison
mc_range_j2 = range((N_M+int(N_M/2))*N_pyr,(1+N_M+int(N_M/2))*N_pyr) # different activation MC for comparison
syns_zipped = list(zip(model.S_REC.i, model.S_REC.j))

all_same_indices = []
all_diff_indices = []

for i_syn, (s, t) in enumerate(syns_zipped):
    if s in mc_range_i:
        if t in mc_range_j1:
            all_same_indices.append(i_syn)
        if t in mc_range_j2:
            all_diff_indices.append(i_syn)

# default spike monitors
spikemon = model.add_spikemon()
basmon = model.add_basmon()

# custom state monitors
synmon_mc_1 = StateMonitor(
    model.S_REC, variables=['w'], 
    record=all_same_indices
)
synmon_mc_2 = StateMonitor(
    model.S_REC, variables=['w'],
    record=all_diff_indices
)
# TODO create add_tracemonitor method in network class or TraceMonitor subclass of StateMonitor
tracemon = StateMonitor(model.REC, variables=model.REC_TRACES, record=[mc_range_i[0], mc_range_j1[0], mc_range_j2[0]])

syn_tracemon_s1 = StateMonitor(model.S_REC, variables=model.S_REC_TRACES+['w'], record=all_same_indices[0])
syn_tracemon_s2 = StateMonitor(model.S_REC, variables=model.S_REC_TRACES+['w'], record=all_diff_indices[0])

# bias state monitors
recorded_biases = np.ndarray.flatten(np.array([mc_range_j1, mc_range_j2]))
biasmon = model.add_statemon(variables=['beta'], record=recorded_biases)

for m in [synmon_mc_1, synmon_mc_2, tracemon, syn_tracemon_s1, syn_tracemon_s2]:
    model.add_monitor(m, m.name)

# show minicolumns that will be studied later on
# fig, ax = plt.subplots()
# synapses.plot_connectivity(
#     ax, model.S_REC, model.N,
#     colors=[(mc_range_i, mc_range_j1,[0,1,0]),(mc_range_i,mc_range_j2,[1,0,0])],
#     aspect='equal'
#     )  
# plt.show()

namespace = model.namespace
defaultclock.dt = namespace['t_sim']
t_stim, t_isi = [namespace[s] for s in ['t_stim', 't_isi']]
t_init, t_end = 100*ms, 100*ms

# calculating eps from total number of timesteps
pattern_list = stils.get_orthogonal_patterns(model.N_H, model.N_M)
# pattern_list = stils.PatternList(pattern_list.patterns[0:1])

# print(",".join([str(p) for p in pattern_list.patterns]))

t_total = get_total_time(t_init, t_stim, t_isi, t_end, N_batches, len(pattern_list.patterns))
model.init_traces(model='zero_weight')
model.namespace['tau_p'] = t_total

# calling train_n_epochs runs the simulation
stims, t_total = cue_n_epochs(
    model, t_init, t_stim, t_isi, t_end,
    pattern_list,
    n_batches=N_batches, shuffle_patterns=True
)

pt_dict = stils.get_pattern_time_dict(pattern_list, stims)

model.save_traces(f'./data/fast-slow/trained_{N_H}_{N_M}_{N_pyr}.data')

# PLOTS

for n_pattern in range(len(pattern_list.patterns)):
    ax = plt.subplot(2, 3, n_pattern+1)
    trains.get_active_freqs_per_batch(
        ax,
        spikemon, n_pattern, model.N_M, model.N_pyr, pattern_list, pt_dict
    )
    ax.set_xlabel('Batch')
    ax.set_ylabel('Spiking Frequency')
    ax.set_title(f'Pattern {n_pattern+1}')
plt.show()

composite.plot_training_protocol(
    model, basmon, spikemon,
    [
        (synmon_mc_1, all_same_indices, 'green', 'co-active neurons'),
        (synmon_mc_2, all_diff_indices, 'red', 'competing neurons')
    ],
    N_batches, t_total, t_div=second,
    pt_dict=pt_dict
)

plt.title('weight trajectories')
# plt.savefig(f'{fig_save_path}/training_protocol_ampa.png')
plt.show()

# composite.plot_bias_trajectory(
#     model, spikemon, biasmon,
#     [np.array(mc_range_j1), np.array(mc_range_j2)], 
#     t_total, second, pt_dict
# )
# plt.savefig(f'{fig_save_path}/')
# plt.savefig(f'./figs/bias_' + suffix + '.png')
# plt.show()

# AMPA EXAMPLE COACTIVE TRACES
ax4 = composite.plot_traces(
    mc_range_i[0], mc_range_j1[0],
    spikemon, tracemon, syn_tracemon_s1, i_syn=0,
    t_div=second
)
plt.title('NMDA coactive trace example')
plt.show()

# AMPA EXAMPLE COMPETING TRACES
ax4 = composite.plot_traces(
    mc_range_i[0], mc_range_j2[0],
    spikemon, tracemon, syn_tracemon_s2, i_syn=0,
    t_div=second
)
plt.title('NMDA competing trace example')
plt.show()


# ax5 = composite.plot_traces(
#     diff_i, diff_j,
#     spikemon, tracemon, syn_tracemon_s2, model.S_REC,
#     t_div=second
# )
# plt.savefig(f'./figs/traces_competing_' + suffix + '.png')
# plt.show()

# AMPA weight matrix
fig, ax = plt.subplots()
im = synapses.plot_weights(ax, model.S_REC, model.N)
fig.colorbar(im, ax=ax)
plt.title('BCPNN weight matrix')
plt.show()