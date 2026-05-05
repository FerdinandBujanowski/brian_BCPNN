# SIMULATION PURPOSE: Show inter-/intra MC dynamics upon stimulation

from brian2 import *
sys.path.append("./")
from brian_bcpnn.networks import TwoSynTypeNetwork
from brian_bcpnn.models.bujanowski_2026.fiebig_params import fiebig_equations, fiebig_namespace
from brian_bcpnn.plot import trains, synapses, composite
from brian_bcpnn.stim_protocols.train_protocol import cue_n_epochs, get_total_time
import brian_bcpnn.utils.stim_utils as stils
import brian_bcpnn.utils.synapse_utils as syls

# simulate only one minicolumn (inside one hypercolumn)
N_H = 6
N_M = 2
N_pyr = 30
N_BA = 4


modded_namespace = fiebig_namespace
# modded_namespace['r_bg'] = 0*Hz
model = TwoSynTypeNetwork(N_H, N_M, N_pyr=N_pyr, N_BA=N_BA, namespace=fiebig_namespace, eqs=fiebig_equations)

# fig, ax = plt.subplots(nrows=2, ncols=3)
# synapses.plot_connectivity(ax[0, 0], model.S_REC, model.N)
# synapses.plot_connectivity(ax[0, 1], model.S_NMDA, model.N)
# synapses.plot_connectivity(ax[0, 2], model.S_MC, model.N)
# synapses.plot_connectivity(ax[1, 0], model.S_PB, N_i=model.N, N_j=model.N_BA_total)
# synapses.plot_connectivity(ax[1, 1], model.S_BP, N_j=model.N, N_i=model.N_BA_total)
# plt.show()

model.namespace['K_AMPA'] = 0
model.namespace['K_NMDA'] = 0

# MONITORS
basmon = model.add_basmon()
spikemon = model.add_spikemon()

# find neurons in minicolumn that are postsynaptically connected to neuron 0
j_list = [j for i,j in zip(model.S_MC.i, model.S_MC.j) if i == 0]
j_list.append(0)
# add voltage monitor
volmon = model.add_statemon(variables=['V_m'], record=j_list)

defaultclock.dt = model.namespace['t_sim']
model.init_traces(model='paper')

pattern_list = stils.get_orthogonal_patterns(model.N_H, model.N_M)
stims, t_total = stils.train_patterns_protocol(pattern_list, 50*ms, 50*ms, 100*ms, 100*ms)
model.namespace['stim_ta'] = stils.stim_times_to_timed_array(stims, t_total, model.N_H, model.N_M)

# run for a while before doing anything
model.run(t_total)

# plot voltages
fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': (1, 1, 5)})
trains.get_full_train(ax1, basmon, model.N_BA_total, t_total,c='b')
trains.get_full_train(ax2, spikemon, model.N, t_total)
ax3.plot(volmon.t/ms, volmon[0].V_m/mV, label='neuron 0')
ax3.plot(volmon.t/ms, volmon[j_list[1]].V_m/mV, label='connected neuron')
ax3.set_ylabel('Voltage (mV)')
ax3.set_xlabel('Time/ms')
ax3.legend()
ax3.grid()
plt.show()