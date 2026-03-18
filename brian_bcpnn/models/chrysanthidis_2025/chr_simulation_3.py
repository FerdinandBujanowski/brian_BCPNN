from brian2 import *
sys.path.append("./")
from brian_bcpnn.networks import ChrysanthidisNetwork
from brian_bcpnn.plot import trains, synapses
from brian_bcpnn.stim_protocols.train_protocol import train_n_epochs

# prefs.codegen.target = 'numpy'
# prefs.codegen.loop_invariant_optimisations = False
# np.seterr(all='raise')

N_hyper = 4
N_mini = 2
N_pyr = 30
N_basket = 4
N_batches = 10

mc_range_i = range(N_pyr) # home MC for comparison
mc_range_j1 = range(2*N_pyr,3*N_pyr) # same activation MC for comparison
mc_range_j2 = range(3*N_pyr,4*N_pyr) # different activation MC for comparison

model = ChrysanthidisNetwork(
    N_hyper, N_mini, N_pyr=N_pyr, N_basket=N_basket, 
    filepath='data/chr/stable_init_test.data', N_poisson=2
)
model.namespace['eps'] = 1/200000

spikemon = model.add_spikemon()
synmon = model.add_synmon(['w'], model.S_REC[0:N_pyr,2*N_pyr:4*N_pyr])
basmon = model.add_basmon()

# show minicolumns that will be studied later on
fig, ax = plt.subplots()
synapses.plot_connectivity(
    ax, model.S_REC, model.N,
    colors=[(mc_range_i, mc_range_j1,[0,1,0]),(mc_range_i,mc_range_j2,[1,0,0])],
    aspect='equal'
    )
plt.show()

namespace = model.namespace
defaultclock.dt = namespace['t_sim']
t_stim, t_isi = [namespace[s] for s in ['t_stim', 'T_stim']]
# TODO put these into param file(s)
t_init, t_end = 100*ms, 200*ms
# calling train_n_epochs runs the simulation
t_total = train_n_epochs(
    model, t_init, t_stim, t_isi, t_end,
    n_batches=N_batches
)

# PLOTS
fig, [ax0, ax1, ax2] = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': (1, 4, 4)})
trains.get_full_train(ax0, basmon, model.N_basket_total, t_total, t_div=second, c='b')
ax0.set_ylabel('# BA neuron')
trains.get_full_train(ax1, spikemon, model.N, t_total, t_div=second)
ax1.set_ylabel('# PYR neuron')

syns_zipped = list(zip(model.S_REC.i, model.S_REC.j))
all_same_synapses = [(int(i),int(j)) for i,j in syns_zipped if i in mc_range_i and j in mc_range_j1]
all_diff_synapses = [(int(i),int(j)) for i,j in syns_zipped if i in mc_range_i and j in mc_range_j2]

synapses.plot_weight_trajectory(
    ax2, model.S_REC, synmon,
    all_same_synapses, 
    t_div=second, c='green', label='co-active neurons'
)
synapses.plot_weight_trajectory(
    ax2, model.S_REC, synmon,
    all_diff_synapses,
    t_div=second, c='red', label='competing neurons'
)
ax2.grid()
ax2.legend()
plt.title(f'{N_batches} batches, t_total={t_total}')
plt.show()