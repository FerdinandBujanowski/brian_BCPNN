# SIMULATION PURPOSE: load trained parameters, turn off plasticity (K=0) 
# and observe model behavior when cueing incomplete patterns
# (still with 100% of active PYR neurons per active MC)

from brian2 import *
sys.path.append("./")
from brian_bcpnn.networks import ChrysanthidisNetwork
from brian_bcpnn.plot import trains, synapses, composite
from brian_bcpnn.stim_protocols.train_protocol import cue_n_epochs, get_total_time
import brian_bcpnn.utils.stim_utils as stils

N_H = 10
N_M = 2
N_pyr = 30
N_BA = 4
N_batches = 2
N_poisson = 1

b_trained = True
filepath = f'data/chr/trained/trained_{N_H}_hc_{N_poisson}_p_longer_isi.data' if b_trained else f'data/chr/stable_init_eps_{N_H}.data'
model = ChrysanthidisNetwork(
    N_H, N_M, N_pyr=N_pyr, N_BA=N_BA, 
    filepath=filepath, N_poisson=N_poisson
)

# DEFAULT MONITORS
spikemon = model.add_spikemon()
basmon = model.add_basmon()
# ..

# CUSTOM MONITORS
# ..

namespace = model.namespace
defaultclock.dt = namespace['t_sim']
t_stim = 50 * ms # cue time
t_isi = 500 * ms
t_init, t_end = 2000*ms, 2000*ms

# Set eps
model.namespace['eps'] = defaultclock.dt/get_total_time(t_init, t_stim, t_isi, t_end, N_batches)
print(model.namespace['eps'])

# Freeze plasticity
model.namespace['K'] = 0

# get orthogonal patterns (assuming model was previously trained on orth patterns!)
pattern_list = stils.get_orthogonal_patterns(model.N_H, model.N_M)
incomplete_patterns = stils.get_incomplete_patterns(pattern_list, int(0.6*N_H)) # 3 out of 5 MCs turned on
print(incomplete_patterns)

stims, t_total = cue_n_epochs(
    model, t_init, t_stim, t_isi, t_end,
    patterns=incomplete_patterns,
    n_batches=N_batches
)
pt_dict = stils.get_pattern_time_dict(incomplete_patterns, stims)
# print(pt_dict)
# print(", ".join([str(stim) for stim in stims]))

fig, [ax0, ax1] = plt.subplots(2, 1, sharex=True,
                                     gridspec_kw={'height_ratios': (1, 5)})
composite.plot_ba_pyr_trains(ax0, ax1, model, basmon, spikemon, t_total, pt_dict=pt_dict)
ax1.set_xlabel(f'Time/{second}')
plt.show()