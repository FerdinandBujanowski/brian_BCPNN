# SIMULATION PURPOSE: Train N_P random (non-correlated patterns)

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

N_P = 10 # amount of random patterns

model = TwoSynTypeNetwork(N_H, N_M, N_pyr=N_pyr, N_BA=N_BA, namespace=fiebig_namespace, eqs=fiebig_equations)

# default spike monitors
spikemon = model.add_spikemon()
basmon = model.add_basmon()

namespace = model.namespace
defaultclock.dt = namespace['t_sim']
t_stim, t_isi = [namespace[s] for s in ['t_stim', 't_isi']]
t_init, t_end = 100*ms, 100*ms

# calculating eps from total number of timesteps
# pattern_list = stils.get_random_patterns(model.N_H, model.N_M, N_P)
pattern_list = stils.patterns_from_txt('10_patterns.txt')
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
# exit()

t_total = get_total_time(t_init, t_stim, t_isi, t_end, N_batches, len(pattern_list.patterns))
model.init_traces(model='zero_weight')
model.namespace['tau_p'] = t_total

# TURN OFF RECURRENCE
model.namespace['b_recurrence'] = 0

# TODO this is just for debugging pt_dict
# stims, t_total = stils.train_patterns_protocol(
#         pattern_list,
#         t_init, t_stim, t_isi, t_end,
#         N_batches, shuffle_patterns=True
#     )

# calling train_n_epochs runs the simulation
stims, t_total = cue_n_epochs(
    model, t_init, t_stim, t_isi, t_end,
    pattern_list,
    n_batches=N_batches, shuffle_patterns=True
)

pt_dict = stils.get_pattern_time_dict(pattern_list, stims)

# print(pt_dict)

# exit()
model.save_traces(f'./data/random_patterns/trained_{N_H}_{N_M}_{N_pyr}_overlap_{N_P}_random_{N_batches}b.data')

# PLOTS

# composite spike train
fig, ax = plt.subplots()
composite.plot_ba_pyr_as_one(ax, model, basmon, spikemon, t_total=t_total, pt_dict=pt_dict, t_div=second)
ax.set_xlabel('Time/s')
plt.show()

# minicolumn average weight matrix
fig, ax = plt.subplots()
im = synapses.plot_weight_matrix_averages(ax, model)
fig.colorbar(im, ax=ax)
plt.title('Average minicolumn weights')
plt.show()