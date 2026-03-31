# SIMULATION PURPOSE: load trained parameters, turn off plasticity (K=0) 
# and observe model behavior on activating imperfect patterns

from brian2 import *
sys.path.append("./")
from brian_bcpnn.networks import ChrysanthidisNetwork
from brian_bcpnn.plot import trains, synapses, composite
from brian_bcpnn.stim_protocols.train_protocol import train_n_epochs, get_total_time
import brian_bcpnn.utils.stim_utils as stils

N_hyper = 5
N_mini = 2
N_pyr = 30
N_basket = 4
N_batches = 0

b_trained = True
filepath = 'data/chr/trained/trained_5_hc_2_p.data' if b_trained else 'data/chr/stable_init_eps_5.data'
model = ChrysanthidisNetwork(
    N_hyper, N_mini, N_pyr=N_pyr, N_basket=N_basket, 
    filepath=filepath, N_poisson=2
)

# DEFAULT MONITORS
spikemon = model.add_spikemon()
basmon = model.add_basmon()
# ..

# CUSTOM MONITORS
# ..

namespace = model.namespace
defaultclock.dt = namespace['t_sim']
t_stim = 0*ms
t_isi = 0*ms
t_init, t_end = 500*ms, 500*ms

# Freeze plasticity
model.namespace['K'] = 0

# get orthogonal patterns (assuming model was previously trained on orth patterns!)
pattern_list = stils.get_orthogonal_patterns(model.N_hyper, model.N_mini)

pattern = pattern_list.patterns[0]
# for i, p in enumerate([0, 0.3, 0.7, 1]):
#     plt.subplot(2, 2, i+1)
#     plt.scatter(range(model.N), model.turn_on_imperfect(pattern, noise_percentage=p), alpha=0.2)
# plt.show()

# turn on imperfect pattern
model.turn_on_imperfect(pattern, noise_percentage=1, turn_on_other=True)

stims, t_total = train_n_epochs(
    model, t_init, t_stim, t_isi, t_end,
    pattern_list,
    n_batches=N_batches
)

fig, [ax0, ax1] = plt.subplots(2, 1, sharex=True,
                                     gridspec_kw={'height_ratios': (1, 5)})
composite.plot_ba_pyr_trains(ax0, ax1, model, basmon, spikemon, t_total)
plt.show()