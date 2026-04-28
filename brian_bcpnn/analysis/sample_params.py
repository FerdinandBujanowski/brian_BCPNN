from brian2 import *
import matplotlib.pyplot as plt
import pickle

filepath = './data/fast-slow/6_6_30_init.data'

def print_dist(x_mean, x_std, name):
    print(f'{name} distribution: mean={round(x_mean,3)}; std={round(x_std,3)}')

data = None
with open(filepath, 'rb') as f:
    data = pickle.load(f)

p_fast = data['P_fast']
p_fast_mean = np.mean(p_fast)
p_fast_std = np.std(p_fast)
print_dist(p_fast_mean, p_fast_std, 'Fast Pre/Post')

p_slow = data['P_slow']
p_slow_mean = np.mean(p_slow)
p_slow_std = np.std(p_slow)
print_dist(p_slow_mean, p_slow_std, 'Slow Pre/Post')

p_syn_fast = data['P_syn_fast']
p_syn_fast_mean = np.mean(p_syn_fast)
p_syn_fast_std = np.std(p_syn_fast)
print_dist(p_syn_fast_mean, p_syn_fast_std, 'Fast Synaptic')

p_syn_slow = data['P_syn_slow']
p_syn_slow_mean = np.mean(p_syn_slow)
p_syn_slow_std = np.std(p_syn_slow)
print_dist(p_syn_slow_mean, p_syn_slow_std, 'Slow Synaptic')

plt.hist(p_fast, label='fast', density=True, alpha=0.7)
plt.hist(p_slow, label='slow', density=True, alpha=0.7)
plt.hist(p_syn_fast, label='fast syn', density=True, alpha=0.7)
plt.hist(p_syn_fast, label='fast syn', density=True, alpha=0.7)
plt.legend()
plt.title("P trace distributions")
plt.show()

# SAMPLE BOTH TO GET WEIGHT DIST
min_num = 0.0001
w_samples = []
for i in range(1000):
    w_samples.append(
        np.log(
            max(min_num, np.random.normal(p_syn_fast_mean, p_syn_fast_std)) /
            (max(min_num, np.random.normal(p_fast_mean, p_fast_std))*max(min_num, np.random.normal(p_fast_mean, p_fast_std)))
        )
    )

w_mean = np.mean(w_samples)
w_std = np.std(w_samples)
print_dist(w_mean, w_std, 'Reconstructed weight')
plt.hist(w_samples, density=True)
plt.show()

# TRY TO RECONSTRUCT P_SYN FROM W AND P_FAST DISTRIBUTIONS
p_syn_fast_reconstr = []
for i in range(1000):
    w_sample = np.random.normal(w_mean, w_std)
    p_sample_1 = max(min_num, np.random.normal(p_fast_mean, p_fast_std))
    p_sample_2 = max(min_num, np.random.normal(p_fast_mean, p_fast_std))

    current_reconstr = (10**w_sample) * p_sample_1 * p_sample_2
    p_syn_fast_reconstr.append(current_reconstr)

reconstr_mean = np.mean(p_syn_fast_reconstr)
reconstr_std = np.std(p_syn_fast_reconstr)
print_dist(reconstr_mean, reconstr_std, 'Reconstructed synaptic P trace')
plt.hist(p_syn_fast, label='syn P', density=True, alpha=0.5)
plt.hist(p_syn_fast_reconstr, density=True, label='Reconstr syn P', alpha=0.5)
plt.show()

# TRY TO RECONSTRUCT P_fast FROM W AND P_Syn DISTRIBUTIONS
p_fast_reconstr = []
for i in range(1000):
    w_sample = np.random.normal(w_mean, w_std)
    p_syn_fast_sample = max(min_num, np.random.normal(p_syn_fast_mean, p_syn_fast_std))

    current_reconstr = sqrt(p_syn_fast_sample/(10**w_sample))
    p_fast_reconstr.append(current_reconstr)

reconstr_mean = np.mean(p_fast_reconstr)
reconstr_std = np.std(p_fast_reconstr)
print_dist(reconstr_mean, reconstr_std, 'Reconstructed fast P trace')
plt.hist(p_fast, label='fast P', density=True, alpha=0.5)
plt.hist(p_fast_reconstr, density=True, label='Reconstr fast P', alpha=0.5)
plt.show()