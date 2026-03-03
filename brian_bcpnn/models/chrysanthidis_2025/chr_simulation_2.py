from brian2 import *
sys.path.append("./")
from brian_bcpnn.networks import ChrysanthidisNetwork
from brian_bcpnn.models.chrysanthidis_2025.chr_params import chr_namespace
from brian_bcpnn.plot import trains

prefs.codegen.target = 'numpy'
prefs.codegen.loop_invariant_optimisations = False
np.seterr(all='raise')

defaultclock.dt = chr_namespace['t_sim']

N_hyper = 64
N_mini = 1
model = ChrysanthidisNetwork(N_hyper, N_mini)

spikemon = SpikeMonitor(model.REC)
model.add_monitor(spikemon, 'spikemon')

model.REC.b_cond[:] = 'int(i<32)'
tfinal = 2000 * ms
model.run(tfinal)

fig, ax = plt.subplots()
trains.get_full_train(ax, spikemon, model.N, tfinal, x_label='time (ms)')
plt.show()