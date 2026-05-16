# SIMULATION PURPOSE: Train N_P random (non-correlated patterns)

from brian2 import *
sys.path.append("./")
from brian_bcpnn.networks import TwoSynTypeNetwork
from brian_bcpnn.models.bujanowski_2026.fiebig_params import fiebig_namespace, fiebig_equations
from brian_bcpnn.plot import trains, synapses, composite
from brian_bcpnn.stim_protocols.train_protocol import cue_n_epochs, get_total_time
import brian_bcpnn.utils.stim_utils as stils
import brian_bcpnn.utils.synapse_utils as syls

N_H = 9
N_M = 9
N_pyr = 5
N_BA = 2
N_batches = 1

N_P = 20 # amount of random patterns

SERIES = 'B'
BATCH = '5_2'

# UPDATE FIEBIG NAMESPACE PARAMETERS
namespace = fiebig_namespace

namespace['b_recurrence'] = 0 # TURN OFF RECURRENCE

# parameter tests
namespace['p_c_inter_hc'] = 1.0
namespace['p_c_intra_mc'] = 1.0
namespace['G_PB_factor'] = 6
namespace['gain_factor'] = 2.2

model = TwoSynTypeNetwork(N_H, N_M, N_pyr=N_pyr, N_BA=N_BA, namespace=namespace, eqs=fiebig_equations)

# default spike monitors
spikemon = model.add_spikemon()
basmon = model.add_basmon()

namespace = model.namespace
defaultclock.dt = namespace['t_sim']
t_stim, t_isi = [namespace[s] for s in ['t_stim', 't_isi']]
t_init, t_end = 100*ms, 0*ms

# calculating eps from total number of timesteps
# pattern_list = stils.get_random_patterns(model.N_H, model.N_M, N_P)
pattern_list = stils.patterns_from_txt(f'20_random_patterns/tests_{SERIES}/patterns_{SERIES}.txt')
print(",".join([str(p) for p in pattern_list.patterns]))

column_list = []
occ_list = []
for pattern in pattern_list.patterns:
    for coords in pattern.coord_list:
        column_list.append(f'{coords.HC}{coords.MC}')
for h in range(N_H):
    for m in range(N_M):
        occ_list.append(column_list.count(f'{h}{m}'))

unique_occs = sorted(list(np.unique(occ_list)))

plt.bar(unique_occs, [occ_list.count(o) for o in unique_occs], width=0.9)
plt.xlabel('# of occurences')
plt.ylabel('Count')
plt.title(f'MC occurence count in {N_P} random patterns')
plt.show()

# pattern_scores = stils.get_pattern_overlap_counts(pattern_list)
# print(pattern_scores)
# unique_scores = sorted(list(np.unique(pattern_scores)))
# plt.bar(unique_scores, [pattern_scores.count(s) for s in unique_scores], width=0.9)
# plt.xlabel('Overlap Score')
# plt.ylabel('Count')
# plt.show()

t_total = get_total_time(t_init, t_stim, t_isi, t_end, N_batches, len(pattern_list.patterns))
model.init_traces(model='zero_weight')
model.namespace['tau_p'] = t_total

# calling train_n_epochs runs the simulation
stims, t_total = cue_n_epochs(
    model, t_init, t_stim, t_isi, t_end,
    pattern_list,
    n_batches=N_batches, shuffle_patterns=False
)

pt_dict = stils.get_pattern_time_dict(pattern_list, stims)

# print(pt_dict)

# exit()
model.save_traces(f'./data/random_patterns/weights_series_{SERIES}_{BATCH}.data')

# PLOTS

# composite spike train
fig, ax = plt.subplots()
composite.plot_ba_pyr_as_one(ax, model, basmon, spikemon, t_total=t_total, pt_dict=pt_dict, t_div=second)
ax.set_xlabel('Time/s')
plt.show()

# minicolumn average weight matrix
fig, ax = plt.subplots()
im, weights = synapses.plot_weight_matrix_averages(ax, model, return_weights=True)
fig.colorbar(im, ax=ax)
plt.title('Average minicolumn weights')
plt.show()

plt.hist(weights)
plt.xlabel('Average MC-MC weight')
plt.ylabel('Count')
plt.show()