import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
import random

def scale_rewards(dataset):
    min_reward = min(dataset['rewards'])
    max_value = max(dataset['rewards'])
    dataset['rewards'] = [-1 + 2 * (x - min_reward) / (max_value - min_reward) for x in dataset['rewards']]
    return dataset

"""
num_t : number of pairs of trajectories
len_t : length of each trajectory
"""
def generate_pbrl_dataset(dataset, num_t, pbrl_dataset_file_path="", len_t=20):
    if pbrl_dataset_file_path != "" and os.path.exists(pbrl_dataset_file_path):
        pbrl_dataset = np.load(pbrl_dataset_file_path)
        print(f"pbrl_dataset loaded successfully from {pbrl_dataset_file_path}")
        return (pbrl_dataset['t1s'], pbrl_dataset['t2s'], pbrl_dataset['ps'])
    else:
        t1s = np.zeros((num_t, len_t), dtype=int)
        t2s = np.zeros((num_t, len_t), dtype=int)
        ps = np.zeros(num_t)
        for i in range(num_t):
            t1, r1 = get_random_trajectory_reward(dataset, len_t)
            t2, r2 = get_random_trajectory_reward(dataset, len_t)
            
            p = np.exp(r1) / (np.exp(r1) + np.exp(r2))
            t1s[i] = t1
            t2s[i] = t2
            ps[i] = p
        np.savez(pbrl_dataset_file_path, t1s=t1s, t2s=t2s, ps=ps)
        # print(f"saving trajectories...")
        return (t1s, t2s, ps)

def get_random_trajectory_reward(dataset, len_t):
    N = dataset['observations'].shape[0]
    start = np.random.randint(0, N-len_t)
    while np.any(dataset['terminals'][start:start+len_t], axis=0):
        start = np.random.randint(0, N-len_t)
    traj = np.array(np.arange(start, start+len_t))
    reward = np.sum(dataset['rewards'][start:start+len_t])
    return traj, reward

def label_by_trajectory_reward(dataset, pbrl_dataset, num_t, len_t=20, num_trials=1):
    # double checking
    t1s, t2s, ps = pbrl_dataset
    sampled = np.random.randint(low=0, high=num_t, size=(num_t,))
    # print(np.max(t1s))
    t1s_indices = t1s[sampled].flatten()
    t2s_indices = t2s[sampled].flatten()
    # t1s_indices = t1s.flatten()
    # t2s_indices = t2s.flatten()
    ps_sample = ps[sampled]
    mus = bernoulli_trial_one_neg_one(ps_sample)
    if num_trials > 1:
        mus = multiple_bernoulli_trials_one_neg_one(ps_sample, num_trials=num_trials)
    repeated_mus = np.repeat(mus, len_t)
    
    sampled_dataset = dataset.copy()
    sampled_dataset['rewards'] = np.array(sampled_dataset['rewards'])
    sampled_dataset['rewards'][t1s_indices] = repeated_mus
    sampled_dataset['rewards'][t2s_indices] = -1 * repeated_mus

    all_indices = np.concatenate([t1s_indices, t2s_indices])
    sampled_dataset['observations'] = sampled_dataset['observations'][all_indices]
    sampled_dataset['actions'] = sampled_dataset['actions'][all_indices]
    sampled_dataset['next_observations'] = sampled_dataset['next_observations'][all_indices]
    sampled_dataset['rewards'] = sampled_dataset['rewards'][all_indices]
    sampled_dataset['terminals'] = sampled_dataset['terminals'][all_indices]

    return sampled_dataset

def bernoulli_trial_one_neg_one(p):
    mus = torch.bernoulli(torch.from_numpy(p)).numpy()
    return -1 + 2 * mus

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)
    
class LatentRewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim = 64, output_dim = 1, activation = nn.ReLU):
        super().__init__()
        self.multi_layer = mlp([input_dim, hidden_dim, hidden_dim, hidden_dim, 1], activation=activation)
        self.one_layer = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.multi_layer(x)
        x = self.tanh(x)
        # x = self.one_layer(x)
        return x
    
"""
pbrl_dataset          : tuple of  (t1s, t2s, p)
latent_reward_X : (2 * N * num_t * len_t , 23)
mus : (2 * N * num_t * len_t, 1)
"""
def make_latent_reward_dataset(dataset, pbrl_dataset, num_t, len_t=20, num_trials=1):
    t1s, t2s, ps = pbrl_dataset
    # print(t1s.shape, np.max(t1s))
    indices = torch.randint(high=num_t, size=(num_t,))
    t1s_sample = t1s[indices]
    t2s_sample = t2s[indices]
    ps_sample = ps[indices]
    obss = dataset['observations']
    acts = dataset['actions']
    indices = np.concatenate((t1s_sample, t2s_sample), axis = 1)
    indices = np.concatenate(indices)
    obs_values = obss[indices] 
    act_values = acts[indices]
    latent_reward_X = np.concatenate((obs_values, act_values), axis=1)
    
    mus = multiple_bernoulli_trials_zero_one(torch.from_numpy(ps_sample), num_trials=num_trials)
    return torch.tensor(latent_reward_X), mus, indices


def train_latent(dataset, pbrl_dataset, num_berno, num_t, len_t,
                 n_epochs = 1000, patience=5, model_file_path=""):
    X, mus, indices = make_latent_reward_dataset(dataset, pbrl_dataset, num_t=num_t, len_t=len_t, num_trials=num_berno)
    if num_berno > 1:
        mus = torch.stack([1 - mus, mus], dim=1)
    else:
        mus = mus.long()
    dim = dataset['observations'].shape[1] + dataset['actions'].shape[1]
    assert((num_t * 2 * len_t, dim) == X.shape)
    model = LatentRewardModel(input_dim=dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_loss = float('inf')
    current_patience = 0

    print('training...')
    for epoch in range(n_epochs):
        total_loss = 0.0
        latent_rewards = model(X).view(num_t, 2, len_t, -1)
        latent_r_sum = torch.sum(latent_rewards, dim=2)
        p = torch.nn.functional.softmax(latent_r_sum, dim=1)
        loss = criterion(p.view(-1, 2), mus)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss = torch.sum(loss)
        if (epoch+1) % 50 == 0:
            print(f'Epoch {epoch + 1}/{n_epochs}, Total Loss: {total_loss}')
            evaluate_latent_model(model, dataset)
            if total_loss < best_loss:
                best_loss = total_loss
                current_patience = 0
                if model_file_path != "":
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                    }, model_file_path)
            else:
                current_patience += 1

            if current_patience >= patience:
                print(f'early stopping after {epoch + 1} epochs without improvement.')
                break
    return model, indices

def evaluate_latent_model(model, dataset, num_t=10000, len_t = 20):
    with torch.no_grad():
        t1s, t2s, ps = generate_pbrl_dataset(dataset, num_t=num_t)
        X_eval, mu_eval, _ = make_latent_reward_dataset(dataset, (t1s, t2s, ps), num_t)
        latent_rewards = model(X_eval).view(num_t, 2, len_t, -1)
        latent_r_sum = torch.sum(latent_rewards, dim=2)
        latent_p = torch.nn.functional.softmax(latent_r_sum, dim=1)[:,1]
        latent_mus = torch.bernoulli(latent_p).long()

        mus_test_flat = mu_eval.view(-1)
        latent_mus_flat = latent_mus.view(-1)
        assert(mus_test_flat.shape == latent_mus_flat.shape)
        accuracy = accuracy_score(mus_test_flat.cpu().numpy(), latent_mus_flat.cpu().numpy())
        print(f'Accuracy: {accuracy:.4f}')

def predict_and_label_latent_reward(dataset, latent_reward_model, indices):
    with torch.no_grad():
        print('predicting and labeling with reward model...')
        obss = dataset['observations']
        acts = dataset['actions']
        obs_values = obss[indices] 
        act_values = acts[indices]
        latent_reward_X = np.concatenate((obs_values, act_values), axis=1)
        latent_rewards = latent_reward_model(torch.tensor(latent_reward_X))
        sampled_dataset = dataset.copy()
        sampled_dataset['rewards'] = latent_rewards

        sampled_dataset['observations'] = sampled_dataset['observations'][indices]
        sampled_dataset['actions'] = sampled_dataset['actions'][indices]
        sampled_dataset['next_observations'] = sampled_dataset['next_observations'][indices]
        sampled_dataset['rewards'] = latent_rewards.view(-1).numpy()
        sampled_dataset['terminals'] = sampled_dataset['terminals'][indices]
        return sampled_dataset

def load_model(model_file_path, dim):
    model = LatentRewardModel(input_dim=dim)
    checkpoint = torch.load(model_file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    return model, epoch

def plot_reward(dataset):
    sorted_rewards = np.sort(dataset['rewards'][::1000])
    indices = np.arange(len(sorted_rewards))
    plt.bar(indices, sorted_rewards, color='blue', alpha=0.7)
    plt.title('Sorted Rewards as a Bar Chart')
    plt.xlabel('Index')
    plt.ylabel('Sorted Rewards')
    plt.savefig('reward_plot.png')
    print("Number of states:", dataset['terminals'].shape[0])
    print("Number of terminal states:", np.sum(dataset['terminals']))

def generate_pbrl_dataset_no_overlap(dataset, num_t, len_t, pbrl_dataset_file_path=""):
    if pbrl_dataset_file_path != "" and os.path.exists(pbrl_dataset_file_path):
        pbrl_dataset = np.load(pbrl_dataset_file_path)
        print(f"pbrl_dataset loaded successfully from {pbrl_dataset_file_path}")
        # print('max t1', np.max(pbrl_dataset['t1s']))
        return (pbrl_dataset['t1s'], pbrl_dataset['t2s'], pbrl_dataset['ps'])
    else:
        # assuming no terminal states
        t1s = np.zeros((num_t, len_t), dtype=int)
        t2s = np.zeros((num_t, len_t), dtype=int)
        ps = np.zeros(num_t)
        starting_indices = list(range(0, len(dataset['observations'])-len_t+1, len_t))
        # print(len(starting_indices))
        for i in range(num_t):
            t1, r1 = pick_and_calc_reward(dataset, starting_indices, len_t)
            t2, r2 = pick_and_calc_reward(dataset, starting_indices, len_t)
            
            p = np.exp(r1) / (np.exp(r1) + np.exp(r2))
            t1s[i] = t1
            t2s[i] = t2
            ps[i] = p
        np.savez(pbrl_dataset_file_path, t1s=t1s, t2s=t2s, ps=ps)
        return (t1s, t2s, ps)
    
def pick_and_calc_reward(dataset, starting_indices, len_t):
    # print(len(starting_indices))
    while True:
        n0 = random.choice(starting_indices)
        starting_indices.remove(n0)
        if np.sum(dataset['terminals'][n0:n0 + len_t - 1]) == 0:
            break

    ns = np.array(np.arange(n0, n0+len_t))
    r = np.sum(dataset['rewards'][n0:n0+len_t])
    return ns, r

def small_d4rl_dataset(dataset, dataset_size_multiplier=1.0):
    if dataset_size_multiplier == 1.0:
        return dataset
    smaller = dataset.copy()
    n_states = dataset['observations'].shape[0]
    sampled_indices = np.random.choice(n_states, size=int(n_states * dataset_size_multiplier), replace=False)
    smaller['observations'] = smaller['observations'][sampled_indices]
    smaller['actions'] = smaller['actions'][sampled_indices]
    smaller['next_observations'] = smaller['next_observations'][sampled_indices]
    smaller['rewards'] = smaller['rewards'][sampled_indices]
    smaller['terminals'] = smaller['terminals'][sampled_indices]
    return smaller


def label_by_trajectory_reward_multiple_bernoullis(dataset, pbrl_dataset, num_t, len_t=20):
    # double checking
    t1s, t2s, ps = pbrl_dataset
    sampled = np.random.randint(low=0, high=num_t, size=(num_t,))
    t1s_indices = t1s[sampled].flatten()
    t2s_indices = t2s[sampled].flatten()
    # t1s_indices = t1s.flatten()
    # t2s_indices = t2s.flatten()
    ps_sample = ps[sampled]
    mus = multiple_bernoulli_trials_one_neg_one(ps_sample, num_trials=10)
    repeated_mus = np.repeat(mus, len_t)
    
    sampled_dataset = dataset.copy()
    sampled_dataset['rewards'] = np.array(sampled_dataset['rewards'])
    sampled_dataset['rewards'][t1s_indices] = repeated_mus
    sampled_dataset['rewards'][t2s_indices] = -1 * repeated_mus

    all_indices = np.concatenate([t1s_indices, t2s_indices])
    sampled_dataset['observations'] = sampled_dataset['observations'][all_indices]
    sampled_dataset['actions'] = sampled_dataset['actions'][all_indices]
    sampled_dataset['next_observations'] = sampled_dataset['next_observations'][all_indices]
    sampled_dataset['rewards'] = sampled_dataset['rewards'][all_indices]
    sampled_dataset['terminals'] = sampled_dataset['terminals'][all_indices]

    return sampled_dataset

def multiple_bernoulli_trials_one_neg_one(p, num_trials):
    if isinstance(p, np.ndarray):
        p = torch.from_numpy(p)
    mus = torch.zeros_like(p)
    for _ in range(num_trials):
        mus += torch.bernoulli(p).numpy()
    return -1 + 2 * (mus / num_trials)

def multiple_bernoulli_trials_zero_one(p, num_trials):
    if isinstance(p, np.ndarray):
        p = torch.from_numpy(p)
    mus = torch.zeros_like(p)
    for _ in range(num_trials):
        mus += torch.bernoulli(p)
    return mus / num_trials

def label_by_original_rewards(dataset, pbrl_dataset, num_t):
    t1s, t2s, _ = pbrl_dataset
    sampled = np.random.randint(low=0, high=num_t, size=(num_t,))
    t1s_indices = t1s[sampled].flatten()
    t2s_indices = t2s[sampled].flatten()
    
    sampled_dataset = dataset.copy()
    all_indices = np.concatenate([t1s_indices, t2s_indices])

    sampled_dataset['observations'] = sampled_dataset['observations'][all_indices]
    sampled_dataset['actions'] = sampled_dataset['actions'][all_indices]
    sampled_dataset['next_observations'] = sampled_dataset['next_observations'][all_indices]
    sampled_dataset['rewards'] = sampled_dataset['rewards'][all_indices]
    sampled_dataset['terminals'] = sampled_dataset['terminals'][all_indices]
    return sampled_dataset