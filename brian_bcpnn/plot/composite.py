from brian2 import *
import matplotlib.pyplot as plt
import matplotlib as mpl
sys.path.append("./")
from brian_bcpnn.utils.stim_utils import StimTime
from brian_bcpnn.plot import traces, trains, synapses

def plot_traces(
        i, j, spikemon, statemon, synmon, i_syn, mode, t_div=second
):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True,
                                     gridspec_kw={'height_ratios': (1, 2, 2, 3, 3)})

    trains.compare_two_trains(ax1, spikemon, i, j, t_div=t_div)

    traces.plot_z_traces(ax2, statemon, i, j, mode=mode, t_div=t_div)

    traces.plot_e_traces(ax3, statemon, synmon, i_syn, i, j, mode=mode, t_div=t_div)

    traces.plot_p_traces(ax4, statemon, synmon, i_syn, i, j, mode=mode, t_div=t_div)

    ax5.plot(synmon.t/t_div, synmon.w[i_syn], c='k')
    ax5.set_ylabel('weight')
    ax5.set_xlabel(f'Time ({t_div})')
    ax5.grid()
    handles, labels = ax4.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')
    return ax5

def plot_training_protocol(
        model, basmon, spikemon, 
        syn_list:list[tuple[StateMonitor, list[int], str, str]],
        N_batches, t_total, t_div=second,
        pt_dict:dict[str,list[StimTime]]=None
    ):
    fig, [ax0, ax1, ax2] = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': (1, 4, 4)})

    plot_ba_pyr_trains(ax0, ax1, model, basmon, spikemon, t_total=t_total, pt_dict=pt_dict)

    for (synmon, syn_indices, color, label) in syn_list:
        synapses.plot_weight_trajectory(
            ax=ax2,
            synmon=synmon, monitored_indices=syn_indices, 
            t_div=t_div, c=color, label=label
        )
    ax2.grid()
    ax2.legend()
    fig.suptitle(f'{N_batches} batches, t_total={t_total}')
    fig.tight_layout()

def plot_spike_train_with_patterns(ax, spikemon, model, t_total, t_div, pt_dict):
    trains.get_full_train(ax, spikemon, model.N, t_total, t_div=t_div)
    if pt_dict is not None:
        cmap = mpl.colormaps['Greens']
        for i, pattern_key in enumerate(pt_dict.keys()):
            for j, stim_time in enumerate(pt_dict[pattern_key]):
                ax.axvspan(
                    xmin=stim_time.t_start/t_div, xmax=stim_time.t_end/t_div, 
                    color=cmap((1+i)/(len(pt_dict)+1)), alpha=0.3, label=(pattern_key if j==0 else None), zorder=0
                )
                ax.legend()
    ax.set_ylabel('# PYR')

def plot_ba_pyr_trains(ax0, ax1, model, basmon, spikemon, t_total, t_div=second, pt_dict=None):
    trains.get_full_train(ax0, basmon, model.N_BA_total, t_total, t_div=t_div, c='b')
    ax0.set_ylabel('# BA')
    plot_spike_train_with_patterns(ax1, spikemon, model, t_total, t_div, pt_dict)

def plot_bias_trajectory(model, spikemon, biasmon, hc_indices:list[list[int]], t_total, t_div=second, pt_dict=None):
    fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': (2, 2)})
    plot_spike_train_with_patterns(ax1, spikemon, model, t_total, t_div, pt_dict)


    bias_cmap = plt.colormaps['Blues']
    for n, indices_list in enumerate(hc_indices):
        biases = np.zeros(shape=(len(indices_list), len(biasmon.t)))
        for i, n_neuron in enumerate(indices_list):
            biases[i,:] = biasmon[n_neuron].beta

        biases_mean = np.mean(biases, axis=0)
        biases_std = np.std(biases, axis=0)
        current_c = bias_cmap((n+1)/(len(hc_indices)+1))
        m_std = 1.96
        ax2.plot(biasmon.t/t_div, biases_mean, c=current_c, label=f'HC {n}')
        ax2.fill_between(biasmon.t/t_div, biases_mean-m_std*biases_std, biases_mean+m_std*biases_std, color=current_c, alpha=0.5)
    ax2.set_xlabel(f'Time/{t_div}')
    ax2.set_ylabel('bias')
    ax2.legend()
    return (ax1, ax2)