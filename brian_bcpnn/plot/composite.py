from brian2 import *
import matplotlib.pyplot as plt
import matplotlib as mpl
sys.path.append("./")
from brian_bcpnn.utils.stim_utils import StimTime
from brian_bcpnn.plot import traces, trains, synapses

def plot_traces(
        i, j, spikemon, statemon, synmon, synapse, t_div=second
):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True,
                                     gridspec_kw={'height_ratios': (1, 2, 2, 3, 3)})

    trains.compare_two_trains(ax1, spikemon, i, j, t_div=t_div)

    traces.plot_z_traces(ax2, statemon, synmon, synapse, i, j, t_div=t_div)

    traces.plot_e_traces(ax3, statemon, synmon, synapse, i, j, t_div=t_div)

    traces.plot_p_traces(ax4, statemon, synmon, synapse, i, j, t_div=t_div)

    ax5.plot(synmon.t/t_div, synmon[synapse[i,j]].w[0], c='k')
    ax5.set_ylabel('weight')
    ax5.set_xlabel(f'Time ({t_div})')
    ax5.grid()
    handles, labels = ax4.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')
    return ax5

def plot_training_protocol(
        model, basmon, spikemon, 
        syn_list:list[tuple[StateMonitor, list[tuple[int,int,str,str]]]],
        N_batches, t_total, t_div=second,
        pt_dict:dict[str,list[StimTime]]=None
    ):
    fig, [ax0, ax1, ax2] = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': (1, 4, 4)})

    trains.get_full_train(ax0, basmon, model.N_basket_total, t_total, t_div=t_div, c='b')
    ax0.set_ylabel('# BA')
    trains.get_full_train(ax1, spikemon, model.N, t_total, t_div=t_div)
    if pt_dict is not None:
        cmap = mpl.colormaps['Greens']
        for i, pattern_key in enumerate(pt_dict.keys()):
            for j, stim_time in enumerate(pt_dict[pattern_key]):
                ax1.axvspan(
                    xmin=stim_time.t_start/t_div, xmax=stim_time.t_end/t_div, 
                    color=cmap((1+i)/(len(pt_dict)+1)), alpha=0.3, label=(pattern_key if j==0 else None), zorder=0
                )
                ax1.legend()
    ax1.set_ylabel('# PYR')

    for (synmon, syn_tuples, color, label) in syn_list:
        synapses.plot_weight_trajectory(
            ax2, model.S_REC,
            synmon, syn_tuples, 
            t_div=t_div, c=color, label=label
        )
    ax2.grid()
    ax2.legend()
    fig.suptitle(f'{N_batches} batches, t_total={t_total}')
    fig.tight_layout()