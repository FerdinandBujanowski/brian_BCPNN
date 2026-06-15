# SIMULATION PURPOSE: track (VERY) detailed statistics of a Minicolumn subgroup 
# (around 10 neurons) of either initialized or pre-trained network in order to
# identify behavior that a "multi-neuron average" entity should follow to be 
# able to be inserted into the network without changes in global network dynamics

from brian2 import *
from tqdm import tqdm
import pandas as pd
import os
sys.path.append("./")
from brian_bcpnn.networks import ChrysanthidisNetwork
from brian_bcpnn.plot import trains, synapses, composite
from brian_bcpnn.stim_protocols.train_protocol import cue_n_epochs, get_total_time
import brian_bcpnn.utils.stim_utils as stils
from brian_bcpnn.utils.stim_utils import PatternProtocol, StimTime

N_H = 5
N_M = 2
N_pyr = 30
N_BA = 4
N_batches = 0

N_neurons = 10
N_record = 30

b_trained = True
filepath = 'data/chr/trained/trained_5_hc_2_p.data' if b_trained else 'data/chr/stable_init_eps_5.data'
model = ChrysanthidisNetwork(
    N_H, N_M, N_pyr=N_pyr, N_BA=N_BA, 
    filepath=filepath, N_poisson=1
)

# DEFAULT MONITORS
spikemon = model.add_spikemon()
basmon = model.add_basmon()
statemon = model.add_statemon(
    variables=['V_m', 'b_pos_noise', 'b_neg_noise'], record=list(range(N_record))
)
# ..

# CUSTOM MONITORS
# ..

namespace = model.namespace
defaultclock.dt = namespace['t_sim']
t_stim, t_isi = [namespace[s] for s in ['t_stim', 't_isi']]
t_init, t_end = 1000*ms, 0*ms

model.namespace['eps'] = defaultclock.dt/get_total_time(t_init, t_stim, t_isi, t_end, N_batches)
print(model.namespace['eps'])

# Freeze plasticity
model.namespace['K'] = 0

# calling train_n_epochs runs the simulation
pattern_list = stils.get_orthogonal_patterns(model.N_H, model.N_M)
stims, t_total = cue_n_epochs(
    model, t_init, t_stim, t_isi, t_end,
    pattern_list,
    n_batches=N_batches
)

print('Model ran successfully.')

output_path='no_input_stats.csv'
entry_list = list()

for _ in tqdm(range(100)):
    # get subset of n_neurons neurons
    indices_choice = np.random.choice(N_record, size=(N_neurons), replace=False)

    # get individual firing freqs
    firing_freqs = np.array([trains.get_neuron_frequency(spikemon, i, t_stop=t_total)/Hz for i in indices_choice])
    # print(firing_freqs)
    f_mean = round(np.mean(firing_freqs), 2)
    f_std = round(np.std(firing_freqs), 2)
    f_sum = np.sum(firing_freqs)

    # fig, ax = plt.subplots()
    # trains.get_full_train(ax, spikemon, model.N, t_total)
    # plt.show()

    # print(f_mean, f_std)
    # print(f_sum)
    
    voltage_matrix = np.zeros(shape=(N_neurons, len(statemon.t)))

    for i, choice_index in enumerate(indices_choice):
        voltage_matrix[i,:] = statemon.V_m[choice_index]/mV
    voltage_argmax = np.argmax(voltage_matrix, axis=0)
    n_switches = 0
    for t in range(len(statemon.t)-1):
        if voltage_argmax[t] != voltage_argmax[t+1]:
            n_switches += 1
    # print(f'{n_switches} switches in total.')

    t_range = range(len(statemon.t))
    max_b_pos = np.zeros(shape=(len(statemon.t)))
    max_b_neg = np.zeros(shape=max_b_pos.shape)
    for t in t_range:
        max_b_pos[t] = statemon.b_pos_noise[indices_choice[voltage_argmax[t]]][t]
        max_b_neg[t] = statemon.b_neg_noise[indices_choice[voltage_argmax[t]]][t]

    n_pos_event = sum(max_b_pos > 0)
    n_neg_event = sum(max_b_neg > 0)

    # print(n_pos_event, n_neg_event)
    entry_list.append({
        'n_total': model.N, 'n_mng': N_neurons, 
        'f_mean': f_mean, 'f_std': f_std, 'f_sum': f_sum,
        'n_pos_event': n_pos_event, 'n_neg_event': n_neg_event,
        'n_switches': n_switches
    })

# write to file
df = pd.DataFrame(entry_list)
df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)

# cmap = colormaps['Greens']
# fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True,
#                                      gridspec_kw={'height_ratios': (1, 3, 3)})
# trains.get_full_train(ax1, spikemon, model.N, t_total, t_div=second, to=N_neurons)
# for i in range(N_neurons):
#     ax2.plot(statemon.t/second, statemon.V_m[i]/mV, alpha=0.3, c='b')
#     ax3.plot(statemon.t/second, statemon.b_pos_noise[i], alpha=0.3, c='g')
#     ax3.plot(statemon.t/second, statemon.b_neg_noise[i], alpha=0.3, c='r')
# ax2.plot(statemon.t/second, voltage_matrix.max(axis=0), c='b', label='max')
# ax2.set_ylabel('Membrane Voltage (mV)')
# ax2.legend()

# ax3.plot(statemon.t/second, max_b_pos, c='g', label='max neuron pos input')
# ax3.plot(statemon.t/second, max_b_neg, c='r', label='max neuron neg input')
# ax3.legend()
# ax3.set_xlabel(f'Time/{second}')

# plt.show()