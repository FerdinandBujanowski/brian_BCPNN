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

N_H = 9
N_M = 9
N_pyr = 15
N_BA = 2
N_batches = 1

defaultclock.dt = 0.1*ms
t_start = 50 * ms
t_isi = 100 * ms
t_stim = 50 * ms
t_end = 0 * ms

SERIES = 'B'
BATCH = '15_4'

fp = f'./data/random_patterns/weights_series_{SERIES}_{BATCH}.data'
pattern_list = stils.patterns_from_txt(f'20_random_patterns/tests_{SERIES}/patterns_{SERIES}.txt')
N_patterns = len(pattern_list.patterns)

t_total = get_total_time(t_start, t_stim, t_isi, t_end, N_batches, N_patterns)

# calculate overlaps BEFORE DISTORTION
pattern_overlaps = stils.get_pattern_overlap_counts(pattern_list)

output_path=f'20_random_patterns/tests_{SERIES}/stats_{BATCH}.csv'

N_dist = 3

# UPDATE FIEBIG NAMESPACE PARAMETERS
namespace = fiebig_namespace
namespace['kappa'] = 0 # turn off plasticity
namespace['tau_p'] = t_total
namespace['b'] = 30 * pA # make attractors last longer

namespace['p_c_inter_hc'] = 0.75 # here it doesn't matter - it needs to be correctly put during training
namespace['p_c_intra_mc'] = 0.5

namespace['G_PB_factor'] = 2
namespace['gain_factor'] = 1

N_runs = 200 # make it 200 after testing
for _ in tqdm(range(N_runs)):

    entry_list = list()

    # distort patterns
    pattern_choice = stils.PatternList([np.random.choice(pattern_list.patterns)])
    distorted_pattern_list = stils.distort_patterns(pattern_choice, N_M, N_dist)
    print(",".join([str(p) for p in pattern_choice.patterns]))
    print(",".join([str(p) for p in distorted_pattern_list.patterns]))

    model = TwoSynTypeNetwork(N_H, N_M, N_pyr=N_pyr, N_BA=N_BA, namespace=namespace, eqs=fiebig_equations, filepath=fp)

    basmon = model.add_basmon()
    spikemon = model.add_spikemon()

    stims, t_total = cue_n_epochs(
        model, t_start, t_stim, t_isi, t_end,
        distorted_pattern_list, n_batches=N_batches, shuffle_patterns=False, b_neg=False
    )
    pt_dict = stils.get_pattern_time_dict(distorted_pattern_list, stims)
    print(pt_dict)

    completion_list = spils.eval_pattern_completion(
        model, spikemon, pattern_choice, pt_dict, t_isi
    )

    for i_pattern, (eval_pattern, eval_nonpattern) in completion_list:
        current_data_point = {
            'N_dist': N_dist, 
            'reconstr_success': eval_pattern,
            'others_silent': eval_nonpattern,
            'original_pattern': str(pattern_choice.patterns[0]),
            'distorted_pattern': str(distorted_pattern_list.patterns[0])
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