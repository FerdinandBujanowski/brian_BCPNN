import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

df = pd.read_csv('no_input_stats.csv')

n_total = 'n_total'
n_mng = 'n_mng'
f_mean = 'f_mean'
f_std = 'f_std'
f_sum = 'f_sum'
n_pos_event = 'n_pos_event'
n_neg_event = 'n_neg_event'
n_switches = 'n_switches'

mng_sorted = sorted(df[n_mng].unique())
mng_space = np.linspace(min(mng_sorted), max(mng_sorted), 1000)
# print(df[df[n_mng] == 5][n_pos_event])

def plot_var_f_mng(ax, var_name, color, label=None):
    var_mean = np.array([df[df[n_mng] == i_mng][var_name].mean() for i_mng in mng_sorted])
    var_std = np.array([df[df[n_mng] == i_mng][var_name].std() for i_mng in mng_sorted])

    ax.plot(mng_sorted, var_mean, c=color, label=label)
    ax.fill_between(mng_sorted, var_mean-var_std, var_mean+var_std, color=color, alpha=0.3)

    return var_mean, var_std

def plot_regression(ax, reg):
    pass

fig, ax = plt.subplots()
mean_pos_event, std_pos_event = plot_var_f_mng(ax, n_pos_event, 'g', 'pos input')
mean_neg_event, std_neg_event = plot_var_f_mng(ax, n_neg_event, 'r', 'neg input')

plt.xlabel('n_mng')
plt.ylabel('mean event number')
plt.legend()
plt.show()

fig, ax = plt.subplots()
# mean_switches, std_switches = plot_var_f_mng(ax, n_switches, color='b', label='# switches')
# switches_reg = linregress(mng_sorted, mean_switches)
# ax.plot(mng_space, mng_space*switches_reg.slope+switches_reg.intercept, c='r', ls='--')
# print('Switches Regression:', (switches_reg.slope, switches_reg.intercept, switches_reg.rvalue))

mean_spike_sum, std_spike_sum = plot_var_f_mng(ax, f_sum, color='k', label='MNG spike sum')
spike_sum_reg = linregress(mng_sorted, mean_spike_sum)
ax.plot(mng_space, mng_space*spike_sum_reg.slope+spike_sum_reg.intercept, c='r', ls='--')
print('Spike Sum Regression:', (spike_sum_reg.slope, spike_sum_reg.intercept, spike_sum_reg.rvalue))
plt.xlabel('n_mng')
plt.ylabel('count')
plt.legend()
plt.show()

# plt.plot(mng_sorted, mean_switches / mean_spike_sum)
# plt.show()