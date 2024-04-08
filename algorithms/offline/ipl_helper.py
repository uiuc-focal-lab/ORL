import numpy as np
from pbrl import multiple_bernoulli_trials_zero_one

def save_preference_dataset(dataset, dpref, dpref_name, num_t, len_t):
    t1s, t2s, ps = dpref
    mus = multiple_bernoulli_trials_zero_one(ps, num_trials=1)
    mus = 1.0 - mus
    states = dataset['observations']
    state_space = dataset['observations'].shape[-1]
    actions = dataset['actions']
    actoin_space = dataset['actions'].shape[-1]
    out = {
        'obs_1': np.zeros((num_t, len_t, state_space)),
        'obs_2': np.zeros((num_t, len_t, state_space)),
        'action_1': np.zeros((num_t, len_t, actoin_space)),
        'action_2': np.zeros((num_t, len_t, actoin_space)),
        'label': np.zeros((num_t,))
    }

    for i in range(num_t):
        start1 = t1s[i, 0]
        start2 = t2s[i, 0]
        for j in range(len_t):
            out['obs_1'][i, j] = states[start1+j]
            out['obs_2'][i, j] = states[start2+j]
            out['action_1'][i, j] = actions[start1+j]
            out['action_2'][i, j] = actions[start2+j]  
        out['label'][i] = mus[i]
    
    np.savez(f'CORL/saved/ipl_dataset/{dpref_name}.npz', **out)
    print(f'Saved preference dataset to CORL/saved/ipl_dataset/{dpref_name}')
    print(f'Shape of obs_1: {out["obs_1"].shape}')
    print(f'Shape of labels: {out["label"].shape}')
