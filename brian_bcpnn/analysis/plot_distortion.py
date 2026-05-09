import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("./")
from brian_bcpnn.utils.stim_utils import pattern_string_to_tuple_list

ND = 'N_dist'
# NO = 'N_overlaps' # doesn't exist yet
RS = 'reconstr_success'
OP = 'original_pattern'
OP_i = 'op_index'
DP = 'distorted_pattern'
MC_OP = 'max_common_OP'
MC_OP_i = 'max_common_OP_i'
MC_DP = 'max_common_DP'
MC_DP_i = 'max_common_DP_i'

# PATH = 'distortion_stats.csv'
BATCH = 3
PATH = f'20_random_patterns/stats_{BATCH}.csv'

df = pd.read_csv(PATH)

success_length = len(df[df[RS] == True])
success_rate = success_length / len(df)
print(f'{success_length}/{len(df)} successfully reconstructed patterns (success rate of {round(success_rate*100, 2)}%).')

# turn all original patterns into sorted list of int tuples (instead of strings)
df[OP] = [pattern_string_to_tuple_list(s) for s in df[OP]]
# do the same for all distorted patterns
df[DP] = [pattern_string_to_tuple_list(s) for s in df[DP]]

# create list of unique original patterns
original_patterns:list[list[tuple[int,int]]] = list(np.unique(df[OP]))

HC = 9
MC = 9

def get_mc_occurence(H, M, original_patterns=original_patterns):
    total = 0
    for pattern in original_patterns:
        for (h, m) in pattern:
            if h==H and m==M:
                total += 1
    return total

all_mc_occurences = dict()
for h in range(HC):
    for m in range(MC):
        all_mc_occurences[(h,m)] = get_mc_occurence(h, m)

def get_pattern_occurence_score(pattern, all_mc_occurences=all_mc_occurences):
    return sum([all_mc_occurences[t] for t in pattern])

all_pattern_scores = [get_pattern_occurence_score(p) for p in original_patterns]
# print(all_pattern_scores)

# turn all original pattern tuples into indices pointing to original pattern list
pattern_indices = [original_patterns.index(s) for s in df[OP]]
df[OP_i] = pattern_indices
unique_pattern_indices = np.array(list(np.unique(pattern_indices)))

# plot histogram of pattern index distribution in dataset
pattern_counts = [pattern_indices.count(i) for i in range(20)]
min_pattern_count = min(pattern_counts)
plt.bar(range(20), pattern_counts)
plt.xlabel('Pattern index')
plt.ylabel('Count in dataset')
plt.xticks(np.arange(0, 20, 5))
plt.show()

# create new column: original pattern occurence score
df['OP_occ'] = [get_pattern_occurence_score(original_patterns[i]) for i in df[OP_i]]

# create new column: distorted pattern occurence score
df['DP_occ'] = [get_pattern_occurence_score(p) for p in df[DP]]

df['occ_diff'] = df['DP_occ'] - df['OP_occ']

def get_max_mc_occ(pattern):
    return max([get_mc_occurence(h,c) for h,c in pattern])

def get_min_mc_occ(pattern):
    return min([get_mc_occurence(h,c) for h,c in pattern])

# column: max minicolumn occurence in pattern (for OP and DP)
df['max_MC_occ'] = [get_max_mc_occ(p) for p in df[OP]]
df['max_MC_occ_DP'] = [get_max_mc_occ(p) for p in df[DP]]
df['min_MC_occ'] = [get_min_mc_occ(p) for p in df[OP]]
df['min_MC_occ_DP'] = [get_min_mc_occ(p) for p in df[DP]]

# TODO create new columns: maximum minicolumns in common with another single pattern (plus index of said pattern)
max_in_common = []
max_in_common_ids = []
max_in_common_times = []

max_common_per_pattern = dict()
for op in df[OP]:
    current_max = 0
    current_max_times = 0
    current_index = 0
    for pattern in original_patterns:
        if pattern != op:
            common = sum([1 for t1, t2 in zip(pattern, op) if t1==t2])
            if common > current_max:
                current_max = common
                current_index = original_patterns.index(pattern)
                current_max_times = 1
            elif common == current_max:
                current_max_times += 1

    max_in_common.append(current_max)
    max_in_common_ids.append(current_index)
    max_in_common_times.append(current_max_times)

    original_index = original_patterns.index(op)
    if original_index not in max_common_per_pattern.keys():
        max_common_per_pattern[original_index] = current_max

all_max_common = np.array([max_common_per_pattern[k] for k in unique_pattern_indices])
max_common_order = np.argsort(all_max_common)
patterns_ordered_by_max_common = unique_pattern_indices[max_common_order]
ordered_max_common = all_max_common[max_common_order]

df[MC_OP] = max_in_common
df[MC_OP_i] = max_in_common_ids
df['MC_OP_times'] = max_in_common_times

max_in_common_dist = []
max_in_common_dist_ids = []

for dp, op in zip(df[DP], df[OP]):
    current_max = 0
    current_index = 0
    for pattern in original_patterns:
        if pattern != op:
            common = sum([1 for t1, t2 in zip(dp, pattern) if t1 == t2])
            if common > current_max:
                current_max = common
                current_index = original_patterns.index(pattern)
    max_in_common_dist.append(current_max)
    max_in_common_dist_ids.append(current_index)

df[MC_DP] = max_in_common_dist
df[MC_DP_i] = max_in_common_dist_ids

unique_mc_op = np.array(list(np.unique(max_in_common)))
unique_mc_dp = np.array(list(np.unique(max_in_common_dist)))

# plt.bar(unique_mc_op-0.2, [max_in_common.count(c) for c in unique_mc_op], label='original patterns', width=0.4)
# plt.bar(unique_mc_dp+0.2, [max_in_common_dist.count(c) for c in unique_mc_dp], label='distorted patterns', width=0.4)
# plt.xlabel('maximum # of common MCs')
# plt.ylabel('Count')
# plt.xticks(np.arange(min(min(unique_mc_op), min(unique_mc_dp)), max(max(unique_mc_op), max(unique_mc_dp))+1, 1))
# plt.legend()
# plt.show()

# TODO new column: closest pattern from distorted pattern is original pattern (yes or no) 


# CREATE SPLIT BETWEEN SUCCESSFUL AND UNSUCCESSFUL TRIALS
df_no_pattern_tuples = df.drop(columns=[OP, DP])
df_success = df_no_pattern_tuples[df[RS] == True]
print(df_success['MC_OP_times'].describe())

df_no_success = df_no_pattern_tuples[df[RS] == False]
print(df_no_success['MC_OP_times'].describe())

def sample_and_plot_success_rate(prop, ax, df=df_no_pattern_tuples, pattern_length=20, N_iter=30, sample_size=min_pattern_count):
    all_prop = []
    for i_pattern in unique_pattern_indices:
        all_prop.append(int(df[df[OP_i]==i_pattern][prop].iloc[0]))

    all_prop = np.array(all_prop)
    prop_order = np.argsort(all_prop)
    patterns_ordered_by_prop = unique_pattern_indices[prop_order]
    ordered_prop = all_prop[prop_order]

    sampled_matrix = np.zeros(shape=(N_iter, pattern_length))
    for n in range(N_iter):
        for i_p, pattern_id in enumerate(patterns_ordered_by_prop):
            pattern_subset = df[df[OP_i]==pattern_id]
            subset_sample = pattern_subset.sample(sample_size)
            success_rate = len(subset_sample[subset_sample[RS] == True])/sample_size
            # if success_rate == 0:
            #     print(original_patterns[list(unique_pattern_indices).index(pattern_id)])
            sampled_matrix[n, i_p] = success_rate

    sampled_mean = np.mean(sampled_matrix, axis=0)
    sampled_std = np.std(sampled_matrix, axis=0)
    (sampled_matrix[:,0])
    ax.errorbar(range(20), sampled_mean, sampled_std, marker='o', ls=' ')
    ax.set_xticks(range(20), ordered_prop)
    ax.set_ylabel('Reconstruction success (%)')
    ax.set_xlabel(f'Patterns (ordered by {prop})')
    ax.grid()
    # plt.show()

fig, axs = plt.subplots(3, 4)
sample_and_plot_success_rate('op_index', axs[0, 0])
sample_and_plot_success_rate('OP_occ', axs[0, 1])
sample_and_plot_success_rate('DP_occ', axs[0, 2])

sample_and_plot_success_rate('max_common_OP', axs[1, 0])
sample_and_plot_success_rate('max_common_DP', axs[1, 1])
sample_and_plot_success_rate('MC_OP_times', axs[1, 2])

sample_and_plot_success_rate('occ_diff', axs[1, 3])

sample_and_plot_success_rate('max_MC_occ', axs[2, 0])
sample_and_plot_success_rate('min_MC_occ', axs[2, 1])
sample_and_plot_success_rate('max_MC_occ_DP', axs[2, 2])
sample_and_plot_success_rate('min_MC_occ_DP', axs[2, 3])

# fig.tight_layout()
plt.show()

scatter_op_occ = []
scatter_max_common = []
success_rates = []
for p_i in unique_pattern_indices:
    # get slice of df of this pattern
    df_slice = df[df[OP_i] == p_i]
    
    # get current OP occ and max_common
    scatter_op_occ.append(int(df_slice['OP_occ'].iloc[0]))
    scatter_max_common.append(int(df_slice[MC_OP].iloc[0]))

    # compute success rate
    success_rates.append(len(df_slice[df_slice[RS] == True])/len(df_slice))

cmap = plt.colormaps['viridis']
plt.scatter(scatter_op_occ, scatter_max_common, c=cmap(success_rates))
plt.colorbar()
plt.show()


# for n_o in unique_n_overlaps:
#     df_current_n_o = df[df[NO] == n_o]
#     success_f_dist = []
#     for n_dist in unique_N_dist:
#         df_current_n_dist = df_current_n_o[df_current_n_o[ND] == n_dist]
#         df_success = df_current_n_dist[df_current_n_dist[RS] == True]
#         success_f_dist.append(len(df_success)/len(df_current_n_dist))
#     plt.plot(unique_N_dist, success_f_dist, label=f'{n_o} overlaps', marker='o')
# plt.legend()
# plt.xlabel('# distortions')
# plt.ylabel('Rate of reconstruction success')
# plt.show()