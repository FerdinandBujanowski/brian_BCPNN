# SIMULATION PURPOSE: load previously learnt weights of non-overlapping patterns and test completion ability

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

fp = f'./data/fast-slow/trained_6_6_30_overlap_1_1b.data'
model = TwoSynTypeNetwork(N_H, N_M, N_pyr=N_pyr, N_BA=N_BA, namespace=fiebig_namespace, eqs=fiebig_equations, filepath=fp)

model.namespace['kappa'] = 0

basmon = model.add_basmon()
spikemon = model.add_spikemon()

defaultclock.dt = model.namespace['t_sim']
t_start = 100 * ms
t_isi = 200 * ms
t_stim = 50 * ms
t_end = 100 * ms
N_batches = 1

full_pattern_list = stils.get_orthogonal_patterns(model.N_H, model.N_M)
pattern_list = stils.get_incomplete_patterns(full_pattern_list, 3)

t_total = get_total_time(t_start, t_stim, t_isi, t_end, N_batches, len(pattern_list.patterns))
model.namespace['tau_p'] = t_total
model.namespace['b'] = 30 * pA # to make attractors last shorter

stims, t_total = cue_n_epochs(
    model, t_start, t_stim, t_isi, t_end,
    pattern_list, n_batches=N_batches
)
pt_dict = stils.get_pattern_time_dict(pattern_list, stims)

fig, [ax0, ax1] = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': (1, 1)})
composite.plot_ba_pyr_as_one(ax0, model, basmon, spikemon, t_total=t_total, pt_dict=pt_dict, t_div=second)
trains.plot_all_minicolumn_activations(ax1, model, spikemon, full_pattern_list, stims, t_total, 10*defaultclock.dt)
ax1.set_xlabel(f'Time/{second}')
plt.show()