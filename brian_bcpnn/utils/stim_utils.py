from brian2 import *
from math import gcd
from dataclasses import dataclass

def add_time(t_total, new_time):
    return t_total+new_time, new_time

def gcd_list(l, o=None):
    match len(l):
        case 0:
            if o is None:
                return 0
            return 0
        case 1:
            if o is None:
                return l[0]
            return gcd(l[0], o)
        case _:
            if o is None:
                return gcd_list(l[2:], gcd(l[0], l[1]))
            return gcd_list(l[1:], gcd(l[0], o))

@dataclass
class ColumnCoords:
    HC: int
    MC: int
    def __str__(self):
        return f'[{self.HC};{self.MC}]'
@dataclass
class StimProtocol:
    coords: ColumnCoords
    t_start: Quantity # in ms
    t_stop: Quantity # in ms
    def __str__(self):
        return f'({str(self.coords)}, {self.t_start}, {self.t_stop})'
@dataclass
class Pattern:
    coord_list: list[ColumnCoords]
    def __str__(self):
        return '['+', '.join([str(coord) for coord in self.coord_list])+']'
@dataclass
class PatternList:
    patterns: list[Pattern]
    def __str__(self):
        return '['+', '.join([str(pattern) for pattern in self.patterns])+']'
    def subset(self,i,j=None):
        if j is not None:
            return PatternList(self.patterns[i:j])
        return PatternList([self.patterns[i]])

def get_orthogonal_patterns(N_H, N_M) -> PatternList:
    return PatternList([Pattern([ColumnCoords(h, m) for h in range(N_H)]) for m in range(N_M)])

def train_patterns_protocol(
        pattern_list: PatternList, 
        t_init:Quantity, t_stim:Quantity, t_isi:Quantity, t_end:Quantity,
        n_batches:int=1
    ) -> tuple[list[StimProtocol], Quantity]:
    stims = []
    current_time = t_init
    for batch in range(n_batches):
        for i_pattern, pattern in enumerate(pattern_list.patterns):
            for coords in pattern.coord_list:
                stims.append(StimProtocol(coords, current_time, current_time+t_stim))
            current_time += t_stim
            if i_pattern < (len(pattern_list.patterns) - 1):
                current_time += t_isi

        if batch < (n_batches-1):
            current_time += t_isi
        else:
            current_time += t_end
    return stims, current_time

def stim_times_to_timed_array(stims: list[StimProtocol], t_total:Quantity, N_H:int, N_M:int):
    times = {int(t_total/ms)}
    hc_list = []
    mc_list = []
    for stim in stims:
        times.add(int(stim.t_start/ms))
        times.add(int(stim.t_stop/ms))
        hc_list.append(stim.coords.HC)
        mc_list.append(stim.coords.MC)

    stim_dt = gcd_list(list(times))*ms
    n_time_steps = int(t_total/stim_dt)

    stim_array = np.zeros(shape=(N_H*N_M,n_time_steps),dtype=int32)
    for i, t in enumerate(range(0, int(t_total/ms), int(stim_dt/ms))):
        for stim in stims:
            if stim.t_start/ms <= t and stim.t_stop/ms > t:
                stim_array[stim.coords.HC*N_M+stim.coords.MC, i] = 1

    # print(stim_array.T)
    return TimedArray(stim_array.T, dt=stim_dt)

# test_protocol = [
#     StimProtocol(ColumnCoords(0, 0), 250*ms, 350*ms),
#     StimProtocol(ColumnCoords(1, 1), 300*ms, 400*ms)
#     ]
# ta = stim_times_to_timed_array(test_protocol, 5*second, 4, 2)