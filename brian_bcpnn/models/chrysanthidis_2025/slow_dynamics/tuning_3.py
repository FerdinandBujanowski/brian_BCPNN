# SIMULATION PURPOSE: Get single intra-MC PYR-BA EPSP to around 0.45 mV (at V_m=-70mV)

from brian2 import *
sys.path.append("./")
from brian_bcpnn.networks import TwoSynTypeNetwork
from brian_bcpnn.models.chrysanthidis_2025.fiebig_params import fiebig_equations, fiebig_namespace
from brian_bcpnn.plot import trains, synapses, composite
from brian_bcpnn.stim_protocols.train_protocol import train_n_epochs, get_total_time
import brian_bcpnn.utils.stim_utils as stils
import brian_bcpnn.utils.synapse_utils as syls

# simulate only one minicolumn (inside one hypercolumn)
N_H = 2
N_M = 1
N_pyr = 30
N_BA = 4


modded_namespace = fiebig_namespace
modded_namespace['r_bg'] = 0*Hz
model = TwoSynTypeNetwork(N_H, N_M, N_pyr=N_pyr, N_BA=N_BA, namespace=fiebig_namespace, eqs=fiebig_equations)

model.namespace['K_AMPA'] = 0
model.namespace['K_NMDA'] = 0

# MONITORS
basmon = model.add_basmon()
spikemon = model.add_spikemon()

# find neurons in Basket layer that are postsynaptically connected to neuron 0
j_list = [j for i,j in zip(model.S_PB.i, model.S_PB.j) if i == 0]
# add PYR voltage monitor
volmon = model.add_statemon(variables=['V_m'], record=0)
# add BA voltage monitor
bas_volmon = StateMonitor(model.BA, variables=['V_m'], record=j_list)
model.add_monitor(bas_volmon, bas_volmon.name)

defaultclock.dt = model.namespace['t_sim']
model.init_traces(model='zero_weight')

# turn off noise input and stim input
# model.namespace['r_bg'] = 0*Hz
# model.namespace['r_stim'] = 0*Hz

# initialize empty timed array 
model.namespace['stim_ta'] = stils.stim_times_to_timed_array([], 1*second, model.N_H, model.N_M)

# run for a while before doing anything
model.run(10*ms, verbose=False)

for set_voltage in range(-80, -60):
    model.BA.V_m[:] = set_voltage * mV

    # cause manual spike for first neuron
    model.REC.V_m[0] = -50*mV

    # run again
    model.run(100*ms, verbose=False)

# plot voltages
fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': (1, 5)})
trains.get_full_train(ax1, basmon, model.N, 2010*ms)
ax2.plot(volmon.t/ms, volmon[0].V_m/mV, label='neuron 0', color='k')
ax2.plot(bas_volmon.t/ms, bas_volmon[j_list[1]].V_m/mV, label='connected BA cell', color='b')
ax2.set_ylabel('Voltage (mV)')
ax2.set_xlabel('Time/ms')
ax2.legend()
ax2.grid()
plt.show()