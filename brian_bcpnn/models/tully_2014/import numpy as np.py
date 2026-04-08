import numpy as np

# Generating 5 patterns (sections) of N=10 bins each.

N=10
s_pre = 
# Section 1: Neurons correlated, random activation/non-activation for both neurons at same time in each bin.
# Both neurons minimally or maximally active/firing in each trial.
# Generating arr of len 10 of 1 and 0 for presynaptic neuron and copying to postsynaptic neuron.
s_pre[0:10] = np.random.randint(0, 2, size=N)
s1post = np.copy(s1pre)
print(s1post)
print(s1pre)

# Section 2: Neurons independent, uniform sampling of active/inactive patterns for both neurons in each trial. 

s2pre = np.random.randint(0, 2, size=N)
s2post = np.random.randint(0,2, size=N)
print(s2pre)
print(s2post)

# Section 3: Randomly distribute pre synaptic? And then set post to 0 when 1 and vice versa?

s3pre = np.random.randint(0, 2, size = N)
s3post = np.zeros(shape=N, dtype=int)
for i in range(len(s3pre)):
    if s3pre[i] == 1:
        s3post[i] = 0
    else:
        s3post[i] = 1

print(s3pre)
print(s3post)

# Section 4: Both muted - inactive

s4pre = np.zeros(N, dtype=int)
s4post = np.zeros(N, dtype=int)

print(s4pre)
print(s4post)


# Section 5: Post muted, activity of pre uniformly sampled and post inactive in all trials.

s5pre = np.random.randint(0,2, size=N)
s5post = np.zeros(N, dtype=int)
print(s5pre)
print(s5post)

