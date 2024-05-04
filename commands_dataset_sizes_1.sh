#!/bin/bash

# binary labels
for seed in 1 2 3
do
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-binary_labels-0.5"             --num_t 49920 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0 --dataset_size_multiplier 0.5
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-binary_labels-0.2"             --num_t 49920 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0 --dataset_size_multiplier 0.2
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-binary_labels-0.1"             --num_t 49920 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0 --dataset_size_multiplier 0.1
done

# latent rewards
for seed in 1 2 3 
do
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-latent_reward-0.5"             --num_t 49920 --len_t 20 --latent_reward 1 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0 --dataset_size_multiplier 0.5
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-latent_reward-0.2"             --num_t 49920 --len_t 20 --latent_reward 1 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0 --dataset_size_multiplier 0.2
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-latent_reward-0.1"             --num_t 49920 --len_t 20 --latent_reward 1 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0 --dataset_size_multiplier 0.1
done