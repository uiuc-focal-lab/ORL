#!/bin/bash

# originals
for seed in 1 2 3
do
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "iql-halfcheetah-medium-expert-original"             --num_t 49920 --len_t 20 --latent_reward 0 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0 
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "hopper-medium-v2"      --out_name "iql-hopper-medium-original"                  --num_t 23000  --len_t 20 --latent_reward 0 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0 
done


# binary labels
for seed in 1 2 3
do
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "iql-halfcheetah-medium-expert-binary_labels"             --num_t 49920 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0 
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "hopper-medium-v2"      --out_name "iql-hopper-medium-binary_labels"                  --num_t 23000  --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0 
done

# latent reward
for seed in 1 2 3
do
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "iql-halfcheetah-medium-expert-latent_reward"             --num_t 49920 --len_t 20 --latent_reward 1 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0 
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "hopper-medium-v2"      --out_name "iql-hopper-medium-latent_reward"                  --num_t 23000  --len_t 20 --latent_reward 1 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0 
done