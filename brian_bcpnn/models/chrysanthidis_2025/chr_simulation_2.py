from brian2 import *
sys.path.append("./")
from brian_bcpnn.networks import ChrysanthidisNetwork
from brian_bcpnn.models.chrysanthidis_2025.chr_params import chr_namespace
from brian_bcpnn.plot import trains, synapses
from brian_bcpnn.utils.synapse_utils import add_time

# prefs.codegen.target = 'numpy'
# prefs.codegen.loop_invariant_optimisations = False
# np.seterr(all='raise')

namespace = chr_namespace
defaultclock.dt = namespace['t_sim']

N_hyper = 2
N_mini = 2
N_pyr = 30
N_basket = 4
model = ChrysanthidisNetwork(N_hyper, N_mini, N_pyr=N_pyr, N_basket=N_basket, namespace=namespace)

spikemon = SpikeMonitor(model.REC)
model.add_monitor(spikemon, 'spikemon')
basmon = StateMonitor(model.BA, variables=['V_m', 'g_ex', 'I_syn'], record=[0])
model.add_monitor(basmon, 'basmon')
recmon = StateMonitor(model.REC, variables=['V_m', 'g_GABA', 'g_AMPA', 'g_BA'], record=[0])
model.add_monitor(recmon, 'recmon')

def plot_rec_synapses(title):
    fig, ax = plt.subplots()
    im = synapses.plot_weights(ax, model.S_REC, model.N, model.N)
    fig.colorbar(im, ax=ax)
    plt.title(title)
    plt.show()

n_batches = 4
t_total, t_no_stim = add_time(0*ms, 500*ms)
model.run(t_no_stim)

# plot_rec_synapses('before stim')

for i in range(n_batches):
    t_total, t_stim = add_time(t_total, model.namespace['t_stim'])
    model.activate_pattern([i, 0] for i in range(N_hyper))
    model.run(t_stim)

    model.turn_off_all()
    t_total, t_stim = add_time(t_total, model.namespace['T_stim'])
    # model.namespace['K'] = 0
    # model.activate_pattern([i, 0] for i in range(N_hyper))
    model.run(t_stim)

    t_total, t_stim = add_time(t_total, model.namespace['t_stim'])
    model.activate_pattern([i, 1] for i in range(N_hyper))
    model.run(t_stim)

    model.turn_off_all()
    t_total, t_stim = add_time(t_total, model.namespace['T_stim'])
    # model.namespace['K'] = 0
    # model.activate_pattern([i, 0] for i in range(N_hyper))
    model.run(t_stim)
    
    if i == (n_batches-1):
        plot_rec_synapses(f'after batch {i+1}')

# print(t_total)
# print(model.S_REC.w)

fig, ax = plt.subplots()
trains.get_full_train(ax, spikemon, model.N, t_total, x_label='time (ms)')
plt.show()

# fig, [ax1, ax2] = plt.subplots(2, 1)
# # lt.plot(basmon.t/ms, basmon.V_m[0]/mV)
# ax1.plot(recmon.t/ms, recmon.g_BA[0]/nS)
# # plt.plot(basmon.t/ms, basmon.g_ex[0]/nS)

# # ax2.plot(basmon.t/ms, basmon.V_m[0]/mV)
# ax2.plot(recmon.t/ms, recmon.V_m[0]/mV)
# plt.show()

