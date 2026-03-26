# SIMULATION PURPOSE: 

from brian2 import *
sys.path.append("./")
from brian_bcpnn.networks import ChrysanthidisNetwork
from brian_bcpnn.models.chrysanthidis_2025.chr_params import chr_namespace
from brian_bcpnn.plot import trains, composite
from brian_bcpnn.utils.synapse_utils import add_time
from tqdm import tqdm

namespace = chr_namespace
defaultclock.dt = namespace['t_sim']

N_hyper = 4
N_mini = 2
N_pyr = 30
N_basket = 4
N_batches = 10
tau_p, t_stim, T_stim = [chr_namespace[s] for s in ['tau_p', 't_stim', 'T_stim']]

mc_range_i = range(N_pyr) # home MC for comparison
mc_range_j1 = range(2*N_pyr,3*N_pyr) # same activation MC for comparison
mc_range_j2 = range(3*N_pyr,4*N_pyr) # different activation MC for comparison

model = ChrysanthidisNetwork(
    N_hyper, N_mini, N_pyr=N_pyr, N_basket=N_basket, 
    namespace=namespace, filepath='data/chr/stable_init.data', N_poisson=2
)

syns_zipped = list(zip(model.S_REC.i, model.S_REC.j))
same_syn = [[int(i),int(j)] for i,j in syns_zipped if i in mc_range_i and j in mc_range_j1]
first_same = same_syn[0]
diff_syn = [[int(i),int(j)] for i,j in syns_zipped if i in mc_range_i and j in mc_range_j2]
first_diff = diff_syn[0]
neuron_indices = list(set(first_same + first_diff))

spikemon = SpikeMonitor(model.REC)
model.add_monitor(spikemon, 'spikemon')
recmon = StateMonitor(model.REC, variables=['Z_j', 'E_j', 'P_j'], record=neuron_indices)
model.add_monitor(recmon, 'recmon')
synmon_same = StateMonitor(model.S_REC, variables=['Z_i', 'E_i', 'P_i', 'E_syn', 'P_syn' ,'w'], record=model.S_REC[first_same[0],first_same[1]])
model.add_monitor(synmon_same, 'synmon_same')
synmon_diff = StateMonitor(model.S_REC, variables=['Z_i', 'E_i', 'P_i', 'E_syn', 'P_syn' ,'w'], record=model.S_REC[first_diff[0],first_diff[1]])
model.add_monitor(synmon_diff, 'synmon_diff')

# fig, ax = plt.subplots()
# synapses.plot_connectivity(
#     ax, model.S_REC, model.N,
#     colors=[
#         ([first_same[0]],[first_same[1]],[0,1,0]),
#         ([first_diff[0]],[first_diff[1]],[1,0,0])
#         ]
#     )
# plt.show()

t_total, t_no_stim = add_time(0*ms, tau_p/2)
model.run(t_no_stim)

for _ in tqdm(range(N_batches)):
    t_total, t_current = add_time(t_total, t_stim)
    model.activate_pattern([i, 0] for i in range(N_hyper))
    model.run(t_current)

    model.turn_off_all()
    t_total, t_current = add_time(t_total, T_stim)
    model.run(t_current)

    t_total, t_current = add_time(t_total, t_stim)
    model.activate_pattern([i, 1] for i in range(N_hyper))
    model.run(t_current)

    model.turn_off_all()
    t_total, t_current = add_time(t_total, T_stim)
    model.run(t_current)

t_total, t_no_stim = add_time(t_total, 2*T_stim)
model.run(t_no_stim)

# Spike Train
fig, ax = plt.subplots()
trains.get_full_train(ax, spikemon, model.N, t_total, t_div=second)
ax.set_xlabel('time (seconds)')
plt.show()

# First Synapse
ax4 = composite.plot_traces(
    first_same[0], first_same[1],
    spikemon, recmon, synmon_same, model.S_REC, t_div=second
)
ax4.set_xticks(np.arange(0, t_total / second, 0.5))
plt.show()