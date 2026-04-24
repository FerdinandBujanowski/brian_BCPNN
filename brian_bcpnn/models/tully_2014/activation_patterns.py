import numpy as np



# Generating 5 patterns (sections) of N=10 bins each.

def activation_lists(N):

    S_pre = np.zeros(N*5, dtype=int)
    S_post = np.zeros(N*5, dtype=int)
    # Section 1: Neurons correlated, random activation/non-activation for both neurons at same time in each bin.
    # Both neurons minimally or maximally active/firing in each trial.
    # Generating arr of len 10 of 1 and 0 for presynaptic neuron and copying to postsynaptic neuron.
    S_pre[0:10] = np.random.randint(0, 2, size=N)
    S_post[0:10] = np.copy(S_pre[0:10])
    #print(S_pre[0:10])
    #print(S_post[0:10])

    # Section 2: Neurons independent, uniform sampling of active/inactive patterns for both neurons in each trial. 

    S_pre[10:20] = np.random.randint(0, 2, size=N)
    S_post[10:20] = np.random.randint(0,2, size=N)
    #print(S_pre[10:20])
    #print(S_post[10:20])

    # Section 3: Randomly distribute pre synaptic? And then set post to 0 when 1 and vice versa?

    S_pre[20:30] = np.random.randint(0, 2, size = N)
    S_post[20:30] = np.zeros(shape=N, dtype=int)
    for i in range(len(S_pre[20:30])):
        if S_pre[20:30][i] == 1:
            S_post[20:30][i] = 0
        else:
            S_post[20:30][i] = 1

    #print(S_pre[20:30])
    #print(S_post[20:30])

    # Section 4: Both muted - inactive

    S_pre[30:40]= np.zeros(N, dtype=int)
    S_post[30:40] = np.zeros(N, dtype=int)

    #print(S_pre[30:40])
    #print(S_post[30:40])

    # Section 5: Post muted, activity of pre uniformly sampled and post inactive in all trials.

    S_pre[40:50] = np.random.randint(0,2, size=N)
    S_post[40:50] = np.zeros(N, dtype=int)
    #print(S_pre[40:50])
    #print(S_post[40:50])
    return S_pre, S_post

#S = activation_lists(10)
#print(S)

#S_pre, S_post = activation_lists(10)

#print(S_pre[0:10], S_pre[10:20], S_pre[20:30], S_pre[30:40], S_pre[40:50])
#print(S_post[0:10], S_post[10:20], S_post[20:30], S_post[30:40], S_post[40:50])