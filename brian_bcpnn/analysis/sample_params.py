from brian2 import *
import matplotlib.pyplot as plt
import pickle

filepath = './data/chr/stable_init_10_2_30.data'

def print_dist(x_mean, x_std, name):
    print(f'{name} distribution: mean={round(x_mean,3)}; std={round(x_std,3)}')

data = None
with open(filepath, 'rb') as f:
    data = pickle.load(f)

p_fast = data['P_fast']
p_mean = np.mean(p_fast)
p_std = np.std(p_fast)
print_dist(p_mean, p_std, 'Fast P trace')

p_syn = data['P_syn_fast']

data['P_slow'] = data['P_fast']
with open(filepath, 'wb') as f:
    pickle.dump(data, f)

p_syn_mean = np.mean(p_syn)
p_syn_std = np.std(p_syn)
print_dist(p_syn_mean, p_syn_std, 'Synaptic P trace')

plt.hist(p_fast, label='fast P', density=True)
plt.hist(p_syn, label='syn P', density=True)
plt.legend()
plt.show()


# SAMPLE BOTH TO GET WEIGHT DIST
min_num = 0.0001
w_samples = []
for i in range(1000):
    w_samples.append(
        np.log(
            max(min_num, np.random.normal(p_syn_mean, p_syn_std)) /
            (max(min_num, np.random.normal(p_mean, p_std))*max(min_num, np.random.normal(p_mean, p_std)))
        )
    )

w_mean = np.mean(w_samples)
w_std = np.std(w_samples)
print_dist(w_mean, w_std, 'Reconstructed weight')
plt.hist(w_samples, density=True)
plt.show()

# TRY TO RECONSTRUCT P_SYN FROM W AND P_FAST DISTRIBUTIONS
p_syn_reconstr = []
for i in range(1000):
    w_sample = np.random.normal(w_mean, w_std)
    p_sample_1 = max(min_num, np.random.normal(p_mean, p_std))
    p_sample_2 = max(min_num, np.random.normal(p_mean, p_std))

    current_reconstr = (10**w_sample) * p_sample_1 * p_sample_2
    p_syn_reconstr.append(current_reconstr)

reconstr_mean = np.mean(p_syn_reconstr)
reconstr_std = np.std(p_syn_reconstr)
print_dist(reconstr_mean, reconstr_std, 'Reconstructed synaptic P trace')
plt.hist(p_syn, label='syn P', density=True, alpha=0.5)
plt.hist(p_syn_reconstr, density=True, label='Reconstr syn P', alpha=0.5)
plt.show()

# TRY TO RECONSTRUCT P_fast FROM W AND P_Syn DISTRIBUTIONS
p_fast_reconstr = []
for i in range(1000):
    w_sample = np.random.normal(w_mean, w_std)
    p_syn_sample = max(min_num, np.random.normal(p_syn_mean, p_syn_std))

    current_reconstr = sqrt(p_syn_sample/(10**w_sample))
    p_fast_reconstr.append(current_reconstr)

reconstr_mean = np.mean(p_fast_reconstr)
reconstr_std = np.std(p_fast_reconstr)
print_dist(reconstr_mean, reconstr_std, 'Reconstructed fast P trace')
plt.hist(p_fast, label='fast P', density=True, alpha=0.5)
plt.hist(p_fast_reconstr, density=True, label='Reconstr fast P', alpha=0.5)
plt.show()