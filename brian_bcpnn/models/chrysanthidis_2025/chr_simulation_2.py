# SIMULATION PURPOSE: Load previously initialised model params (with same architecture)
# and train orthogonal patterns for several batches - save final parameters to file 

from brian2 import *
sys.path.append("./")
from brian_bcpnn.networks import ChrysanthidisNetwork
from brian_bcpnn.plot import trains, synapses, composite
from brian_bcpnn.stim_protocols.train_protocol import train_n_epochs, get_total_time
import brian_bcpnn.utils.stim_utils as stils
import brian_bcpnn.utils.synapse_utils as syls

# prefs.codegen.target = 'numpy'
# prefs.codegen.loop_invariant_optimisations = False
# np.seterr(all='raise')

N_H = 5
N_M = 2
N_pyr = 30
N_BA = 4
N_poisson = 1
N_batches = 2

model = ChrysanthidisNetwork(
    N_H, N_M, N_pyr=N_pyr, N_BA=N_BA, 
    filepath=f'data/chr/stable_init_eps_{N_H}.data', N_poisson=N_poisson
)

# TEST: init 10x2x30 model by sampling traces and weights from 5x2x30 model
# sample_filepath = 'data/chr/stable_init_eps_5.data'
# model = ChrysanthidisNetwork(
#     N_H, N_M, N_pyr, N_BA, N_poisson
# )
# model.sample_params(sample_filepath)

# SYNAPSE INDEXING
mc_range_i = range(N_pyr) # home MC for comparison
mc_range_j1 = range(N_M*N_pyr,(N_M+1)*N_pyr) # same activation MC for comparison
mc_range_j2 = range((2*N_M-1)*N_pyr,2*N_M*N_pyr) # different activation MC for comparison
syns_zipped = list(zip(model.S_REC.i, model.S_REC.j))
all_same_synapses = [(int(i),int(j)) for i,j in syns_zipped if i in mc_range_i and j in mc_range_j1]
all_diff_synapses = [(int(i),int(j)) for i,j in syns_zipped if i in mc_range_i and j in mc_range_j2]
(same_i, same_j) = all_same_synapses[0]
(diff_i, diff_j) = all_diff_synapses[0]

# default spike monitors
spikemon = model.add_spikemon()
basmon = model.add_basmon()

# custom state monitors
synmon_mc_1 = StateMonitor(
    model.S_REC, variables=['w'], 
    # record=syls.get_synapse_indices(model.S_REC, mc_range_i, mc_range_j1)
    record=model.S_REC[min(mc_range_i):max(mc_range_i)+1,min(mc_range_j1):max(mc_range_j1)+1]
)
synmon_mc_2 = StateMonitor(
    model.S_REC, variables=['w'],
    # record=syls.get_synapse_indices(model.S_REC, mc_range_i, mc_range_j2)
    record=model.S_REC[min(mc_range_i):max(mc_range_i)+1,min(mc_range_j2):max(mc_range_j2)+1]
)
# TODO create add_tracemonitor method in network class or TraceMonitor subclass of StateMonitor
tracemon = StateMonitor(model.REC, variables=model.REC_TRACES, record=[same_j, diff_j])
syn_tracemon_s1 = StateMonitor(model.S_REC, variables=model.S_REC_TRACES+['w'], record=syls.get_synapse_indices(model.S_REC, [same_i], [same_j]))
syn_tracemon_s2 = StateMonitor(model.S_REC, variables=model.S_REC_TRACES+['w'], record=syls.get_synapse_indices(model.S_REC, [diff_i], [diff_j]))

# bias state monitors
recorded_biases = np.ndarray.flatten(np.array([mc_range_j1, mc_range_j2]))
biasmon = model.add_statemon(variables=['beta'], record=recorded_biases)

for m in [synmon_mc_1, synmon_mc_2, tracemon, syn_tracemon_s1, syn_tracemon_s2]:
    model.add_monitor(m, m.name)

# show minicolumns that will be studied later on
fig, ax = plt.subplots()
synapses.plot_connectivity(
    ax, model.S_REC, model.N,
    colors=[(mc_range_i, mc_range_j1,[0,1,0]),(mc_range_i,mc_range_j2,[1,0,0])],
    aspect='equal'
    )
plt.show()

namespace = model.namespace
defaultclock.dt = namespace['t_sim']
t_stim, t_isi = [namespace[s] for s in ['t_stim', 't_isi']]
# TODO put these into param file(s)
t_init, t_end = 100*ms, 200*ms

# calculating eps from total number of timesteps
model.namespace['eps'] = defaultclock.dt/get_total_time(t_init, t_stim, t_isi, t_end, N_batches)
print(model.namespace['eps'])
# dt/t_total is the same as 1/n_steps <=> 1/(t_total/dt)

# calling train_n_epochs runs the simulation
pattern_list = stils.get_orthogonal_patterns(model.N_H, model.N_M)
stims, t_total = train_n_epochs(
    model, t_init, t_stim, t_isi, t_end,
    pattern_list,
    n_batches=N_batches
)

pt_dict = stils.get_pattern_time_dict(pattern_list, stims)
# for key in pt_dict.keys():
#     print(f'[{key}]: {[str(i) for i in pt_dict[key]]}')

# model.save_traces(f'data/chr/trained/trained_{N_H}_hc_{N_poisson}_p.data')

# PLOTS
for n_pattern in [0, 1]:
    fig, ax = plt.subplots()
    trains.get_active_freqs_per_batch(
        ax,
        spikemon, n_pattern, model.N_M, model.N_pyr, pattern_list, pt_dict
    )
    plt.xlabel('Batch')
    plt.ylabel('Spiking Frequency')
    plt.title(f'Pattern {n_pattern+1}')
    plt.show()

composite.plot_training_protocol(
    model, basmon, spikemon,
    [
        (synmon_mc_1, all_same_synapses, 'green', 'co-active neurons'),
        (synmon_mc_2, all_diff_synapses, 'red', 'competing neurons')
    ],
    N_batches, t_total, t_div=second,
    pt_dict=pt_dict
)
plt.show()

composite.plot_bias_trajectory(
    model, spikemon, biasmon,
    [np.array(mc_range_j1), np.array(mc_range_j2)], 
    t_total, second, pt_dict
)
plt.show()

ax4 = composite.plot_traces(
    same_i, same_j,
    spikemon, tracemon, syn_tracemon_s1, model.S_REC,
    t_div=second
)
plt.show()

ax5 = composite.plot_traces(
    diff_i, diff_j,
    spikemon, tracemon, syn_tracemon_s2, model.S_REC,
    t_div=second
)
plt.show()

fig, ax = plt.subplots()
im = synapses.plot_weights(ax, model.S_REC, model.N)
fig.colorbar(im, ax=ax)
plt.show()