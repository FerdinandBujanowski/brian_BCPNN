from brian2 import *
sys.path.append("./")
from brian_bcpnn.networks import MAX_PYR
from brian_bcpnn.plot import trains, synapses
import brian_bcpnn.utils.stim_utils as stils

def init_network_params(model, filepath):
    namespace = model.namespace
    defaultclock.dt = namespace['t_sim']

    spikemon = SpikeMonitor(model.REC)
    model.add_monitor(spikemon, 'spikemon')
    
    pre_indices, post_indices = model.S_REC.i, model.S_REC.j
    choice_indices = np.random.choice(list(range(len(pre_indices))), size=30, replace=False)
    chosen_pre = [int(pre_indices[i]) for i in choice_indices]
    chosen_post = [int(post_indices[i]) for i in choice_indices]
    mon_tuples = [(i,j) for i,j in zip(chosen_pre, chosen_post)]
    # print(mon_tuples)
    synmon = StateMonitor(model.S_REC, variables=['w'], record=model.S_REC[chosen_pre,chosen_post])
    w_init_mon = StateMonitor(model.S_REC, variables=['w_init'], record=model.S_REC[chosen_pre[0], chosen_post[0]])
    for m in [synmon, w_init_mon]:
        model.add_monitor(m, m.name)

    statemon = model.add_statemon(['V_m'], record=[0])
    
    # Important first step: freeze plasticity at start of simulation,
    # slowly ease back into it through 'dw_init/dt' in chr_model.py synapse equation

    no_stim = []
    namespace['tau_init'] = 7.5 * second * (MAX_PYR / model.N_pyr)
    t_total = 3*namespace['tau_init']
    print(f'Simulating for {t_total} (tau_init={namespace["tau_init"]})')
    model.namespace['stim_ta'] = stils.stim_times_to_timed_array(no_stim, t_total, model.N_H, model.N_M)
    
    # initialize p-traces and run without plasticity
    original_K = model.namespace['K']
    model.namespace['eps'] = defaultclock.dt/t_total
    # print(model.namespace['eps'])
    # model.namespace['K'] = 0
    model.S_REC.w_init[:] = 1
    model.init_traces()
    # turn on plasticity and run until the weights converge
    # model.namespace['K'] = original_K
    # model.S_REC.w_init[:] = 0
    model.run(t_total)

    # save all important values into file
    model.save_traces(filepath)

    _, [ax1, ax2, ax3] = plt.subplots(1, 3) #gridspec_kw={'width_ratios': (2, 1)})
    trains.get_full_train(ax1, spikemon, model.N, t_total, t_div=second)
    ax1.set_xlabel(f'Time/{second}')
    ax1.set_title('Spike Train')

    freqs = trains.get_spiking_histogram(ax2, spikemon, model.N, t_start=t_total-2*second, t_stop=t_total)
    avg_freq = round(np.mean(freqs), 2)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Neuron Count')
    ax2.set_title(f'Firing Freq Dist (Mean={avg_freq}Hz)')
    # trains.sliding_window_freq(ax1, spikemon, model.N, t_stop=t_total)
    
    synapses.plot_weight_trajectory(ax3, model.S_REC, synmon, mon_tuples, t_div=second)
    ax3.set_title('Avg Weight Trajectory')

    ax4 = ax3.twinx()
    c = 'tab:red'
    ax4.plot(w_init_mon.t/second, w_init_mon[model.S_REC[chosen_pre[0], chosen_post[0]]].w_init[0], label='w_init', color=c, ls='--')
    ax4.tick_params(axis='y', labelcolor=c)
    ax4.set_ylabel('w_init')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(statemon.t/second, statemon.V_m[0]/mV)
    ax.set_title('Example Neuron Membrane Voltage')
    ax.set_xlabel(f'Time/{second}')
    ax.set_ylabel('Voltage/mV')
    plt.show()

    return spikemon

def get_sampled(source, target, ax=None, x_label=None):
    out = np.zeros(shape=(len(target),))
    source_mean = np.mean(source)
    source_std = np.std(source)

    out = np.random.normal(source_mean, source_std, size=(len(target),))

    if ax is not None:
        ax.hist(source)
        ax.set_xlabel(x_label)
        ax.set_ylabel('count')
    return out