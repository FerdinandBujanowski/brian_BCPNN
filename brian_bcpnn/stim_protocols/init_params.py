from brian2 import *
sys.path.append("./")
from brian_bcpnn.networks import CorticalNetwork
from brian_bcpnn.plot import trains, synapses
import brian_bcpnn.utils.stim_utils as stils

def init_network_params(model:CorticalNetwork, filepath, plot_train=True):
    namespace = model.namespace
    defaultclock.dt = namespace['t_sim']

    spikemon = None
    synmon = None
    mon_tuples = None
    if plot_train:
        spikemon = SpikeMonitor(model.REC)
        model.add_monitor(spikemon, 'spikemon')
        
        pre_indices, post_indices = model.S_REC.i, model.S_REC.j
        choice_indices = np.random.choice(list(range(len(pre_indices))), size=30, replace=False)
        chosen_pre = [int(pre_indices[i]) for i in choice_indices]
        chosen_post = [int(post_indices[i]) for i in choice_indices]
        mon_tuples = [(i,j) for i,j in zip(chosen_pre, chosen_post)]
        # print(mon_tuples)
        synmon = StateMonitor(model.S_REC, variables=['w'], record=model.S_REC[chosen_pre,chosen_post])
        model.add_monitor(synmon, 'synmon')
    
    # Important first step: freeze plasticity at start of simulation,
    # slowly ease back into it through 'dw_init/dt' in chr_model.py synapse equation
    # TODO update this with TimedArray

    no_stim = []
    t_total = 20*second
    t_freeze = 5*second
    model.namespace['stim_ta'] = stils.stim_times_to_timed_array(no_stim, t_total, model.N_hyper, model.N_mini)
    
    # initialize p-traces and run without plasticity
    original_K = model.namespace['K']
    eps = 0.25 # model.namespace['eps']
    # model.namespace['K'] = 0
    # model.S_REC.w_init[:] = 1
    model.REC.P_j[:] = eps
    model.S_REC.P_i[:] = eps
    model.S_REC.P_syn[:] = eps ** 2
    model.run(t_freeze)

    # turn on plasticity and run until the weights converge
    model.namespace['K'] = original_K
    model.run(t_total-t_freeze)

    # save all important values into file
    model.save_traces(filepath)

    if plot_train:
        _, [ax1, ax2, ax3] = plt.subplots(1, 3)
        trains.get_full_train(ax1, spikemon, model.N, t_total, t_div=second)
        ax1.set_xlabel(f'Time/{second}')
        ax1.set_title('Spike Train')

        freqs = trains.get_spiking_histogram(ax2, spikemon, model.N, t_start=t_total-2*second, t_stop=t_total)
        avg_freq = round(np.mean(freqs), 2)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Neuron Count')
        ax2.set_title(f'Firing Freq Dist (Mean={avg_freq}Hz)')
        
        # TODO third panel with average weight trajectory
        synapses.plot_weight_trajectory(ax3, model.S_REC, synmon, mon_tuples, t_div=second)
        ax3.set_title('Avg Weight Trajectory')
        plt.show()
    return spikemon

def load_params_test():
    pass