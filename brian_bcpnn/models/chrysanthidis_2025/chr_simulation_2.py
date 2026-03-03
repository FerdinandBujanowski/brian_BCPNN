from brian2 import *
sys.path.append("./")
from brian_bcpnn.networks import ChrysanthidisNetwork
from brian_bcpnn.models.chrysanthidis_2025.chr_params import chr_namespace
from brian_bcpnn.plot import trains

# prefs.codegen.target = 'numpy'
# prefs.codegen.loop_invariant_optimisations = False
# np.seterr(all='raise')

namespace = chr_namespace
defaultclock.dt = namespace['t_sim']

N_hyper = 1
N_mini = 2
N_pyr = 30
N_basket = 4
model = ChrysanthidisNetwork(N_hyper, N_mini, N_pyr=N_pyr, N_basket=N_basket, namespace=namespace)

spikemon = SpikeMonitor(model.REC)
model.add_monitor(spikemon, 'spikemon')

def add_time(t_total, new_time):
    return t_total+new_time, new_time


t_total, t_no_stim = add_time(0*ms, namespace['T_stim'])
model.run(t_no_stim)

t_total, t_stim = add_time(t_total, namespace['t_stim'])
model.activate_pattern([i, 0] for i in range(N_hyper))
model.run(t_stim)

print(t_total)

fig, ax = plt.subplots()
trains.get_full_train(ax, spikemon, model.N, t_total, x_label='time (ms)')
plt.show()