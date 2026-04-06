from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append("./")
from brian_bcpnn.networks import TullyNetwork
from brian_bcpnn.plot import composite
from brian_bcpnn.utils.stim_utils import StimProtocol, ColumnCoords, StimTime
import brian_bcpnn.utils.stim_utils as stils
import brian_bcpnn.utils.synapse_utils as syls

NEW_TAU_P = 1*second
defaultclock.dt = 0.1 * ms
t_total = 10*NEW_TAU_P # 10 * tau_p

# STIMULI
neuron_1_coords = ColumnCoords(0, 0)
neuron_2_coords = ColumnCoords(1, 0)
figure_4_stims = [
    # add stimprotocols..
    # ..
    StimProtocol(neuron_1_coords, StimTime(0*ms, 100*ms)),
    StimProtocol(neuron_2_coords, StimTime(100*ms, 200*ms)),
    StimProtocol(neuron_1_coords, StimTime(300*ms, 400*ms)),
    StimProtocol(neuron_2_coords, StimTime(300*ms, 400*ms))

]

n_iterations = 10
t_array = None
w_array = np.zeros(shape=(n_iterations, int(t_total/defaultclock.dt)))
print(w_array.shape)

# REPEATED SIMULATION
for i in tqdm(range(n_iterations)):
    model = TullyNetwork(verbose=False)
    model.namespace['tau_p'] = NEW_TAU_P
    model.namespace['stim_ta'] = stils.stim_times_to_timed_array(figure_4_stims, t_total, model.N_H, model.N_M)

    # MONITORS
    weightmon = model.add_synmon(variables=['w'], record=True)
    # TODO add statemon for bias

    model.run(t_total)

    w_array[i,:] = weightmon.w[0]
    if t_array is None:
        t_array = weightmon.t/model.namespace['tau_p']
    

# calculate and plot stats for w and bias

w_mean = np.mean(w_array, axis=0)
w_std = np.std(w_array, axis=0)
n_std = 1.96

plt.plot(t_array, w_mean, color='b')
plt.fill_between(t_array, w_mean-n_std*w_std, w_mean+n_std*w_std, color='b', alpha=0.3)

plt.show()