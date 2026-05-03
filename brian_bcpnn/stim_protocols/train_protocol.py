from brian2 import *
sys.path.append("./")
from brian_bcpnn.networks import CorticalNetwork
from brian_bcpnn.plot import trains
import brian_bcpnn.utils.stim_utils as stils

def train_n_epochs(
        model:CorticalNetwork,
        t_init, t_stim, t_isi, t_end,
        patterns:stils.PatternList,
        n_batches=1,
        stim_ta_string='stim_ta',
        shuffle_patterns=False
):
    defaultclock.dt = model.namespace['t_sim']
    
    stims, t_total = stils.train_patterns_protocol(
        patterns,
        t_init, t_stim, t_isi, t_end,
        n_batches, shuffle_patterns=shuffle_patterns
    )

    model.namespace[stim_ta_string] = stils.stim_times_to_timed_array(
        stims, t_total,
        model.N_H, model.N_M
    )
    model.run(t_total)

    return stims, t_total

def get_total_time(t_init, t_stim, t_isi, t_end, n_batches=1, n_patterns=1):
    return (
        + t_init
        + n_patterns * (n_batches*t_stim
        + max(n_batches-1,0)*t_isi)
        + t_end
    )