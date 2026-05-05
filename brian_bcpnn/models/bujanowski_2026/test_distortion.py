# SIMULATION PURPOSE: Generate + save data on completion and error correction capacities of minimally overlapping patterns

from brian2 import *
import pandas as pd
import os

sys.path.append("./")
from brian_bcpnn.networks import TwoSynTypeNetwork
from brian_bcpnn.models.bujanowski_2026.fiebig_params import fiebig_equations, fiebig_namespace
from brian_bcpnn.plot import trains, synapses, composite
from brian_bcpnn.stim_protocols.train_protocol import cue_n_epochs, get_total_time
import brian_bcpnn.utils.stim_utils as stils
import brian_bcpnn.utils.spike_utils as spils
from tqdm import tqdm

fp = f'./data/random_patterns/trained_6_6_30_overlap_10_random_2b.data'

N_H = 6
N_M = 6
N_pyr = 30
N_BA = 4
N_batches = 1

defaultclock.dt = 0.1*ms
t_start = 100 * ms
t_isi = 250 * ms
t_stim = 50 * ms
t_end = 250 * ms

# TODO load patterns from file or other way

# pattern_list = stils.get_orthogonal_patterns(N_H, N_M)
# pattern_list.patterns[3].coord_list[4] = stils.ColumnCoords(4, 4)
# pattern_list.patterns[3].coord_list[5] = stils.ColumnCoords(5, 5)
# pattern_list.patterns[5].coord_list[3] = stils.ColumnCoords(3, 4)
pattern_list = stils.patterns_from_txt('10_patterns.txt')
N_patterns = len(pattern_list.patterns)

t_total = get_total_time(t_start, t_stim, t_isi, t_end, N_batches, N_patterns)

# calculate overlaps BEFORE DISTORTION
pattern_overlaps = stils.get_pattern_overlap_counts(pattern_list)

output_path='distortion_stats_10_random.csv'

N_dist_weight = [1, 2, 3, 3, 1]
N_dist_freq = np.array(N_dist_weight) / sum(N_dist_weight)

N_runs = 1
for _ in tqdm(range(N_runs)):
# for N_dist in tqdm(range(N_M)):
    N_iter = 1
    N_dist = np.random.choice(range(5), p=N_dist_freq)
    print(f'Running {N_iter} simulations for N_dist={N_dist}')
    for _ in range(N_iter):
        entry_list = list()

        # distort patterns
        distorted_pattern_list = stils.distort_patterns(pattern_list, N_M, N_dist)
        # print(",".join([str(p) for p in pattern_list.patterns]))

        model = TwoSynTypeNetwork(N_H, N_M, N_pyr=N_pyr, N_BA=N_BA, namespace=fiebig_namespace, eqs=fiebig_equations, filepath=fp)

        basmon = model.add_basmon()
        spikemon = model.add_spikemon()

        model.namespace['kappa'] = 0
        model.namespace['tau_p'] = t_total
        model.namespace['b'] = 30 * pA # to make attractors last shorter

        stims, t_total = cue_n_epochs(
            model, t_start, t_stim, t_isi, t_end,
            distorted_pattern_list, n_batches=N_batches, shuffle_patterns=False
        )
        pt_dict = stils.get_pattern_time_dict(distorted_pattern_list, stims)
        completion_list = spils.eval_pattern_completion(
            model, spikemon, pattern_list, pt_dict, t_isi
        )

        for i_pattern, completion in completion_list:
            current_data_point = {
                'N_dist': N_dist, 
                'N_overlaps': pattern_overlaps[i_pattern], 
                'reconstr_success': completion,
                'original_pattern': str(pattern_list.patterns[i_pattern]),
                'distorted_pattern': str(distorted_pattern_list.patterns[i_pattern])
            }
            entry_list.append(current_data_point)

        df = pd.DataFrame(entry_list)
        df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
        print(f'Saved {len(entry_list)} datapoints to .csv file')

    # print(entry_list)
    # fig, ax = plt.subplots()
    # composite.plot_ba_pyr_as_one(ax, model, basmon, spikemon, t_total=t_total, pt_dict=pt_dict, t_div=second)
    # ax.set_xlabel(f'Time/{second}')
    # plt.show()