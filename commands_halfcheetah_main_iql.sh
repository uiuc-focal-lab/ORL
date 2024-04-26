#!/bin/bash

# originals
for seed in 1 2 3
do
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "iql-halfcheetah-medium-expert-original"             --num_t 49920 --len_t 20 --latent_reward 0 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "halfcheetah-medium-replay-v2" --out_name "iql-halfcheetah-medium-replay-original"             --num_t 5000  --len_t 20 --latent_reward 0 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "halfcheetah-medium-v2"        --out_name "iql-halfcheetah-medium-original"                    --num_t 24960 --len_t 20 --latent_reward 0 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0

# latent rewards
for seed in 1 2 3 4 5
do
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "iql-halfcheetah-medium-expert-latent_reward"             --num_t 49920 --len_t 20 --latent_reward 1 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "halfcheetah-medium-replay-v2" --out_name "iql-halfcheetah-medium-replay-latent_reward"             --num_t 5000  --len_t 20 --latent_reward 1 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "halfcheetah-medium-v2"        --out_name "iql-halfcheetah-medium-latent_reward"                    --num_t 24960 --len_t 20 --latent_reward 1 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
done

# latent rewards multiple bernoulli
for seed in 1 2 3
do
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "iql-halfcheetah-medium-expert-latent_reward_multi_berno" --num_t 49920 --len_t 20 --latent_reward 1 --bin_label 0 --num_berno 10 --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "halfcheetah-medium-replay-v2" --out_name "iql-halfcheetah-medium-replay-latent_reward_multi_berno" --num_t 5000  --len_t 20 --latent_reward 1 --bin_label 0 --num_berno 10 --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "halfcheetah-medium-v2"        --out_name "iql-halfcheetah-medium-latent_reward_multi_berno"        --num_t 24960 --len_t 20 --latent_reward 1 --bin_label 0 --num_berno 10 --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
done

# binary labels
for seed in 1 2 3 4 5
do
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "iql-halfcheetah-medium-expert-binary_labels"             --num_t 49920 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "halfcheetah-medium-replay-v2" --out_name "iql-halfcheetah-medium-replay-binary_labels"             --num_t 5000  --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "halfcheetah-medium-v2"        --out_name "iql-halfcheetah-medium-binary_labels"                    --num_t 24960 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
done

# binary labels multiple bernoulli
for seed in 1 2 3
do
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "iql-halfcheetah-medium-expert-binary_labels_multi_berno" --num_t 49920 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 10 --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "halfcheetah-medium-replay-v2" --out_name "iql-halfcheetah-medium-replay-binary_labels_multi_berno" --num_t 5000  --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 10 --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "halfcheetah-medium-v2"        --out_name "iql-halfcheetah-medium-binary_labels_multi_berno"        --num_t 24960 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 10 --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
done

done