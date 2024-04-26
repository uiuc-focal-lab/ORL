#!/bin/bash

# originals
for seed in 1 2 3
do
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-expert-v2" --out_name "cql-walker-medium-expert-original"             --num_t 49600 --len_t 20 --latent_reward 0 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-replay-v2" --out_name "cql-walker-medium-replay-original"             --num_t 7000  --len_t 20 --latent_reward 0 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-v2"        --out_name "cql-walker-medium-original"                    --num_t 24800 --len_t 20 --latent_reward 0 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0

# latent rewards
for seed in 1 2 3 4 5
do
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-expert-v2" --out_name "cql-walker-medium-expert-latent_reward"             --num_t 49600 --len_t 20 --latent_reward 1 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-replay-v2" --out_name "cql-walker-medium-replay-latent_reward"             --num_t 7000  --len_t 20 --latent_reward 1 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-v2"        --out_name "cql-walker-medium-latent_reward"                    --num_t 24800 --len_t 20 --latent_reward 1 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
done

# latent rewards multiple bernoulli
for seed in 1 2 3
do
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-expert-v2" --out_name "cql-walker-medium-expert-latent_reward_multi_berno" --num_t 49600 --len_t 20 --latent_reward 1 --bin_label 0 --num_berno 10 --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-replay-v2" --out_name "cql-walker-medium-replay-latent_reward_multi_berno" --num_t 7000  --len_t 20 --latent_reward 1 --bin_label 0 --num_berno 10 --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-v2"        --out_name "cql-walker-medium-latent_reward_multi_berno"        --num_t 24800 --len_t 20 --latent_reward 1 --bin_label 0 --num_berno 10 --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
done

# binary labels
for seed in 1 2 3 4 5
do
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-expert-v2" --out_name "cql-walker-medium-expert-binary_labels"             --num_t 49600 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-replay-v2" --out_name "cql-walker-medium-replay-binary_labels"             --num_t 7000  --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-v2"        --out_name "cql-walker-medium-binary_labels"                    --num_t 24800 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
done

# binary labels multiple bernoulli
for seed in 1 2 3
do
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-expert-v2" --out_name "cql-walker-medium-expert-binary_labels_multi_berno" --num_t 49600 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 10 --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-replay-v2" --out_name "cql-walker-medium-replay-binary_labels_multi_berno" --num_t 7000  --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 10 --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-v2"        --out_name "cql-walker-medium-binary_labels_multi_berno"        --num_t 24800 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 10 --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0
done

done