import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("./")
from brian_bcpnn.utils.stim_utils import pattern_string_to_tuple_list


ND = 'N_dist'
NO = 'N_overlaps'
RS = 'reconstr_success'
OP = 'original_pattern'
DP = 'distorted_pattern'

df = pd.read_csv('distortion_stats.csv')

# turn all original patterns into sorted list of int tuples (instead of strings)
df[OP] = [pattern_string_to_tuple_list(s) for s in df[OP]]
# do the same for all distorted patterns
df[DP] = [pattern_string_to_tuple_list(s) for s in df[DP]]

# create list of unique original patterns
original_patterns = list(np.unique(df[OP]))
HC = 6
MC = 6

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
df[OP] = [original_patterns.index(s) for s in df[OP]]

# create new column: original pattern occurence score
df['OP_occ'] = [get_pattern_occurence_score(original_patterns[i]) for i in df[OP]]

# create new column: distorted pattern occurence score
df['DP_occ'] = [get_pattern_occurence_score(p) for p in df[DP]]

unique_n_overlaps = list(np.unique(df[NO]))
unique_N_dist = sorted(list(np.unique(df[ND])))

for n_o in unique_n_overlaps:
    df_current_n_o = df[df[NO] == n_o]
    success_f_dist = []
    for n_dist in unique_N_dist:
        df_current_n_dist = df_current_n_o[df_current_n_o[ND] == n_dist]
        df_success = df_current_n_dist[df_current_n_dist[RS] == True]
        success_f_dist.append(len(df_success)/len(df_current_n_dist))
    plt.plot(unique_N_dist, success_f_dist, label=f'{n_o} overlaps', marker='o')
plt.legend()
plt.xlabel('# distortions')
plt.ylabel('Rate of reconstruction success')
plt.show()