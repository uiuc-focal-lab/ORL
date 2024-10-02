import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import os
import random
import pandas as pd

# num_t : number of pairs of trajectories
# len_t : length of each trajectory

def scale_rewards(dataset):
    # Scales rewards in the dataset to the range [-1, 1].
    min_reward = min(dataset['rewards'])
    max_value = max(dataset['rewards'])
    dataset['rewards'] = [-1 + 2 * (x - min_reward) / (max_value - min_reward) for x in dataset['rewards']]
    return dataset

def generate_pbrl_dataset(dataset, num_t, pbrl_dataset_file_path="", len_t=20):
    # Generates a PBRL dataset with pairs of trajectories and the probability of preferring the first trajectory.
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
            if np.isnan(p):
                p = float(r1 > r2)
            t1s[i] = t1
            t2s[i] = t2
            ps[i] = p
        np.savez(pbrl_dataset_file_path, t1s=t1s, t2s=t2s, ps=ps)
        print(f"saving trajectories...")
        return (t1s, t2s, ps)

def get_random_trajectory_reward(dataset, len_t):
    # Selects a random trajectory and calculates its reward.
    N = dataset['observations'].shape[0]
    start = np.random.randint(0, N-len_t)
    while np.any(dataset['terminals'][start:start+len_t], axis=0):
        start = np.random.randint(0, N-len_t)
    traj = np.array(np.arange(start, start+len_t))
    reward = np.sum(dataset['rewards'][start:start+len_t])
    return traj, reward

def label_by_trajectory_reward(dataset, pbrl_dataset, num_t, len_t=20, num_trials=1, name=None):
    # Labels the dataset with binary rewards based on trajectory preferences.
    print('labeling with binary reward...')
    t1s, t2s, ps = pbrl_dataset
    t1s_indices = t1s.flatten()
    t2s_indices = t2s.flatten()
    mus = bernoulli_trial_one_neg_one(ps)
    if num_trials > 1:
        mus = multiple_bernoulli_trials_one_neg_one(ps, num_trials=num_trials)
    repeated_mus = np.repeat(mus, len_t)
    new_dataset = dataset.copy()
    new_dataset['rewards'] = np.zeros_like(dataset['rewards'])

    preferred_indices = torch.zeros((num_t * len_t,), dtype=int)
    rejected_indices = torch.zeros((num_t * len_t,), dtype=int)
    for i in range(num_t * len_t):
        if repeated_mus[i] >= 0.5:
            preferred_indices[i] = t1s_indices[i]
            rejected_indices[i] = t2s_indices[i]
        else:
            preferred_indices[i] = t2s_indices[i]
            rejected_indices[i] = t1s_indices[i]

    # Take average in case of repeated trajectories
    index_count = {}
    for i in range(len(t1s_indices)):
        t1s_index = t1s_indices[i]
        t2s_index = t2s_indices[i]
        index_count[t1s_index] = index_count.get(t1s_index, 0) + 1
        index_count[t2s_index] = index_count.get(t2s_index, 0) + 1

    for i in range(len(t1s_indices)):
        t1s_index = t1s_indices[i]
        t2s_index = t2s_indices[i]
        new_dataset['rewards'][t1s_index] += repeated_mus[i] / index_count[t1s_index]
        new_dataset['rewards'][t2s_index] += -1 * repeated_mus[i] / index_count[t2s_index]

    all_indices = np.concatenate([t1s_indices, t2s_indices])
    new_dataset['observations'] = new_dataset['observations'][all_indices]
    new_dataset['actions'] = new_dataset['actions'][all_indices]
    new_dataset['next_observations'] = new_dataset['next_observations'][all_indices]
    new_dataset['rewards'] = new_dataset['rewards'][all_indices]
    new_dataset['terminals'] = new_dataset['terminals'][all_indices]

    rewards = np.array(dataset['rewards'])
    rewards[all_indices] = np.array(new_dataset['rewards'])
    preferred_rewards = rewards[preferred_indices]
    sampled_preferred_rewards = preferred_rewards[torch.randperm(len(preferred_rewards))[:10000]]
    rejected_rewards = rewards[rejected_indices]
    sampled_rejected_rewards = rejected_rewards[torch.randperm(len(rejected_rewards))[:10000]]
    if name:
        df = pd.DataFrame({'preferred_reward': sampled_preferred_rewards, 'rejected_reward': sampled_rejected_rewards})
        df.to_csv(f'saved/sampled_rewards/{name}.csv', index=False)
    return new_dataset

def bernoulli_trial_one_neg_one(p):
    # Performs a Bernoulli trial and maps the result to [-1, 1].
    mus = torch.bernoulli(torch.from_numpy(p)).numpy()
    return -1 + 2 * mus

def mlp(sizes, activation, output_activation=nn.Identity):
    # Creates a multi-layer perceptron (MLP) neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)
    
class LatentRewardModel(nn.Module):
    # Defines a neural network model for latent reward prediction.
    def __init__(self, input_dim, hidden_dim = 64, output_dim = 1, activation = nn.ReLU):
        super().__init__()
        self.multi_layer = mlp([input_dim, hidden_dim, hidden_dim, hidden_dim, output_dim], activation=activation)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.multi_layer(x)
        x = self.tanh(x)
        return x
    
"""
input:
    pbrl_dataset    : tuple of (t1s, t2s, p)
output:
    latent_reward_X : (2 * num_t * len_t, 23)
    mus             : (2 * num_t * len_t, 1)
"""
def make_latent_reward_dataset(dataset, pbrl_dataset, num_t, len_t=20, num_trials=1):
    # Creates a dataset for training the latent reward model.
    t1s, t2s, ps = pbrl_dataset
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

    preferred_indices = torch.zeros((num_t, len_t), dtype=int)
    rejected_indices = torch.zeros((num_t, len_t), dtype=int)
    for i in range(num_t):
        if mus[i] >= 0.5:
            preferred_indices[i] = torch.from_numpy(t1s_sample[i])
            rejected_indices[i] = torch.from_numpy(t2s_sample[i])
        else:
            preferred_indices[i] = torch.from_numpy(t2s_sample[i])
            rejected_indices[i] = torch.from_numpy(t1s_sample[i])
    return torch.tensor(latent_reward_X), mus, indices, preferred_indices.view(-1), rejected_indices.view(-1)

def train_latent(dataset, pbrl_dataset, num_berno, num_t, len_t, name="",
                 n_epochs = 200, patience=5, model_file_path=""):
    # Trains the latent reward model using the PBRL dataset.
    X, mus, indices, preferred_indices, rejected_indices = make_latent_reward_dataset(dataset, pbrl_dataset, num_t=num_t, len_t=len_t, num_trials=num_berno)
    dim = dataset['observations'].shape[1] + dataset['actions'].shape[1]
    assert((num_t * 2 * len_t, dim) == X.shape)
    model = LatentRewardModel(input_dim=dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_loss = float('inf')
    current_patience = 0
    pref_r_sample = None
    rejt_r_sample = None
    
    print('training...')
    for epoch in range(n_epochs):
        total_loss = 0.0
        latent_rewards = model(X).view(num_t, 2, len_t, -1)
        latent_r_sum = torch.sum(latent_rewards, dim=2)
        p = torch.nn.functional.softmax(latent_r_sum, dim=1)
        loss = criterion(p.view(-1, 2)[:,0], mus.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss = torch.sum(loss)
        if (epoch+1) % 25 == 0:
            print(f'Epoch {epoch + 1}/{n_epochs}, Total Loss: {total_loss}')
            training_data = (X, mus)
            pref_r_sample, rejt_r_sample = evaluate_latent_model(model, dataset, training_data, num_t=num_t, preferred_indices=preferred_indices, rejected_indices=rejected_indices)
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
    if name:
        df = pd.DataFrame({'preferred_reward': pref_r_sample.numpy(), 'rejected_reward': rejt_r_sample.numpy()})
        df.to_csv(f'saved/sampled_rewards/{name}.csv', index=False)
    return model, indices

def evaluate_latent_model(model, dataset, training_data, num_t, preferred_indices, rejected_indices, testing_num_t=1000, len_t=20):
    # Evaluates the latent reward model on both training and testing data.
    with torch.no_grad():
        # Training evaluation
        X_train, mu_train = training_data
        if torch.all(mu_train.type(torch.int64) == mu_train):
            latent_rewards_train = model(X_train).view(num_t, 2, len_t, -1)
            latent_r_sum_train = torch.sum(latent_rewards_train, dim=2)
            latent_p_train = torch.nn.functional.softmax(latent_r_sum_train, dim=1)[:,0]
            latent_mu_train = torch.bernoulli(latent_p_train).long()
            
            mu_train_flat = mu_train.view(-1)
            latent_mu_train_flat = latent_mu_train.view(-1)
            assert(mu_train_flat.shape == latent_mu_train_flat.shape)
            train_accuracy = accuracy_score(mu_train_flat.cpu().numpy(), latent_mu_train_flat.cpu().numpy())
            print(f'Train Accuracy: {train_accuracy:.3f}')

        # Testing evaluation
        t1s, t2s, ps = generate_pbrl_dataset_no_overlap(dataset, num_t=testing_num_t, len_t=len_t, save=False)
        X_eval, mu_eval, _, _, _= make_latent_reward_dataset(dataset, (t1s, t2s, ps), testing_num_t)
        latent_rewards = model(X_eval).view(testing_num_t, 2, len_t, -1)
        latent_r_sum = torch.sum(latent_rewards, dim=2)
        latent_p = torch.nn.functional.softmax(latent_r_sum, dim=1)[:,0]
        latent_mus = torch.bernoulli(latent_p).long()

        mus_test_flat = mu_eval.view(-1)
        latent_mus_flat = latent_mus.view(-1)
        assert(mus_test_flat.shape == latent_mus_flat.shape)
        accuracy = accuracy_score(mus_test_flat.cpu().numpy(), latent_mus_flat.cpu().numpy())
        print(f'Test Accuracy: {accuracy:.3f}')

        # Preferred and rejected reward gap
        real_rewards = np.array(dataset['rewards'])

        preferred_indices = preferred_indices.cpu().numpy()
        preferred_obs_values = dataset['observations'][preferred_indices] 
        preferred_act_values = dataset['actions'][preferred_indices]
        true_preferred_rewards = real_rewards[preferred_indices]
        preferred_training_data = np.concatenate((preferred_obs_values, preferred_act_values), axis=1)
        preferred_rewards = model(torch.tensor(preferred_training_data)).view(-1)
        sampled_preferred_rewards = preferred_rewards[torch.randperm(len(preferred_rewards))[:10000]]
        expected_preferred_reward = torch.mean(sampled_preferred_rewards)
        print(f"Expected Reward for preferred (s,a) pairs in the training set: {expected_preferred_reward}")
        print(f"True     Reward for preferred (s,a) pairs in the training set: {np.mean(true_preferred_rewards)}")

        rejected_indices = rejected_indices.cpu().numpy()
        rejected_obs_values = dataset['observations'][rejected_indices]
        rejected_act_values = dataset['actions'][rejected_indices]
        true_rejected_rewards = real_rewards[rejected_indices]
        rejected_training_data = np.concatenate((rejected_obs_values, rejected_act_values), axis=1)
        rejected_rewards = model(torch.tensor(rejected_training_data)).view(-1)
        sampled_rejected_rewards = rejected_rewards[torch.randperm(len(rejected_rewards))[:10000]]
        expected_rejected_reward = torch.mean(sampled_rejected_rewards)
        print(f"Expected Reward for rejected  (s,a) pairs in the training set: {expected_rejected_reward}")
        print(f"True     Reward for rejected  (s,a) pairs in the training set: {np.mean(true_rejected_rewards)}")

        return sampled_preferred_rewards, sampled_rejected_rewards

def predict_and_label_latent_reward(dataset, latent_reward_model, indices):
    # Predicts and labels the dataset with the latent reward model.
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
    # Loads a saved latent reward model from a file.
    model = LatentRewardModel(input_dim=dim)
    checkpoint = torch.load(model_file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    return model, epoch

def generate_pbrl_dataset_no_overlap(dataset, num_t, len_t, reuse_fraction=0.0, reuse_times=0, pbrl_dataset_file_path="", save=True):
    # Generates a PBRL dataset ensuring no overlap between trajectories.
    if pbrl_dataset_file_path != "" and os.path.exists(pbrl_dataset_file_path):
        pbrl_dataset = np.load(pbrl_dataset_file_path)
        print(f"pbrl_dataset loaded successfully from {pbrl_dataset_file_path}")
        return (pbrl_dataset['t1s'], pbrl_dataset['t2s'], pbrl_dataset['ps'])
    else:
        t1s = np.zeros((num_t, len_t), dtype=int)
        t2s = np.zeros((num_t, len_t), dtype=int)
        ps = np.zeros(num_t)
        starting_indices = list(range(0, len(dataset['observations'])-len_t+1, len_t))
        num_reuse = int(num_t * reuse_fraction)
        starting_indices_to_reuse = np.random.choice(starting_indices, num_reuse, replace=False)
        starting_indices_to_reuse = list(np.repeat(starting_indices_to_reuse, reuse_times))
        starting_indices_not_to_reuse = [x for x in starting_indices if x not in starting_indices_to_reuse]

        for i in range(num_t):
            if len(starting_indices_to_reuse):
                t1, r1 = pick_and_calc_reward(dataset, starting_indices_to_reuse, len_t)
            else:
                t1, r1 = pick_and_calc_reward(dataset, starting_indices_not_to_reuse, len_t)
            t2, r2 = pick_and_calc_reward(dataset, starting_indices_not_to_reuse, len_t)
            
            p = np.exp(r1) / (np.exp(r1) + np.exp(r2))
            if np.isnan(p):
                p = float(r1 > r2)
            t1s[i] = t1
            t2s[i] = t2
            ps[i] = p
        if save:
            np.savez(pbrl_dataset_file_path, t1s=t1s, t2s=t2s, ps=ps)
        return (t1s, t2s, ps)
    
def pick_and_calc_reward(dataset, starting_indices, len_t):
    # Picks a starting index and calculates the reward for a trajectory.
    while True:
        n0 = random.choice(starting_indices)
        starting_indices.remove(n0)
        if np.sum(dataset['terminals'][n0:n0 + len_t - 1]) == 0:
            break

    ns = np.array(np.arange(n0, n0+len_t))
    r = np.sum(dataset['rewards'][n0:n0+len_t])
    return ns, r

def small_d4rl_dataset(dataset, dataset_size_multiplier=1.0):
    # Reduces the size of the dataset by a given multiplier.
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

def multiple_bernoulli_trials_one_neg_one(p, num_trials):
    # Performs multiple Bernoulli trials and maps the results to [-1, 1].
    if isinstance(p, np.ndarray):
        p = torch.from_numpy(p)
    mus = torch.zeros_like(p)
    for _ in range(num_trials):
        mus += torch.bernoulli(p).numpy()
    return -1 + 2 * (mus / num_trials)

def multiple_bernoulli_trials_zero_one(p, num_trials):
    # Performs multiple Bernoulli trials and maps the results to [0, 1].
    if isinstance(p, np.ndarray):
        p = torch.from_numpy(p)
    mus = torch.zeros_like(p)
    for _ in range(num_trials):
        mus += torch.bernoulli(p)
    return mus / num_trials

def label_by_original_rewards(dataset, pbrl_dataset, num_t):
    # Labels the dataset with the original rewards.
    t1s, t2s, _ = pbrl_dataset
    t1s_indices = t1s.flatten()
    t2s_indices = t2s.flatten()
    
    sampled_dataset = dataset.copy()
    all_indices = np.concatenate([t1s_indices, t2s_indices])

    sampled_dataset['observations'] = sampled_dataset['observations'][all_indices]
    sampled_dataset['actions'] = sampled_dataset['actions'][all_indices]
    sampled_dataset['next_observations'] = sampled_dataset['next_observations'][all_indices]
    sampled_dataset['rewards'] = sampled_dataset['rewards'][all_indices]
    sampled_dataset['terminals'] = sampled_dataset['terminals'][all_indices]
    return sampled_dataset

def pick_and_generate_pbrl_dataset(dataset, env, num_t, len_t, num_trials=1, allow_overlap=1, reuse_fraction=0.0, reuse_times=0):
    # Picks and generates a PBRL dataset based on the given parameters.
    if allow_overlap and reuse_fraction == 0.0:
        dataset_path = f'saved/pbrl_datasets/pbrl_dataset_{env}_{num_t}_{len_t}_numTrials={num_trials}.npz'
        pbrl_dataset = generate_pbrl_dataset(dataset, num_t=num_t, len_t=len_t, pbrl_dataset_file_path=dataset_path)
    if allow_overlap and reuse_fraction > 0.0:
        dataset_path_reuse = f'saved/pbrl_datasets_reuse/pbrl_dataset_{env}_{num_t}_{len_t}_numTrials={num_trials}_reuse({reuse_fraction}-{reuse_times})'
        pbrl_dataset = generate_pbrl_dataset(dataset, num_t=num_t, len_t=len_t, pbrl_dataset_file_path=dataset_path_reuse, reuse_fraction=reuse_fraction, reuse_times=reuse_times)
    if not allow_overlap and reuse_fraction == 0.0:
        daraset_path_no_overlap = f'saved/pbrl_datasets_no_overlap/pbrl_dataset_{env}_{num_t}_{len_t}_numTrials={num_trials}'
        pbrl_dataset = generate_pbrl_dataset_no_overlap(dataset, num_t=num_t, len_t=len_t, pbrl_dataset_file_path=daraset_path_no_overlap)
    if not allow_overlap and reuse_fraction > 0.0:
        dataset_path_reuse_no_overlap = f'saved/pbrl_datasets_no_overlap_reuse/pbrl_dataset_{env}_{num_t}_{len_t}_numTrials={num_trials}_reuse({reuse_fraction}-{reuse_times})'
        pbrl_dataset = generate_pbrl_dataset_no_overlap(dataset, num_t=num_t, len_t=len_t, pbrl_dataset_file_path=dataset_path_reuse_no_overlap, reuse_fraction=reuse_fraction, reuse_times=reuse_times)

    return pbrl_dataset
