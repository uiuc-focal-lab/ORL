#!/bin/bash

for seed in 1 2 3
do
    # originals
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-original"             --num_t 49920 --len_t 20 --latent_reward 0 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-replay-v2" --out_name "cql-halfcheetah-medium-replay-original"             --num_t 5000  --len_t 20 --latent_reward 0 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-v2"        --out_name "cql-halfcheetah-medium-original"                    --num_t 24960 --len_t 20 --latent_reward 0 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed

    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "hopper-medium-expert-v2" --out_name "cql-hopper-medium-expert-original"             --num_t 48800 --len_t 20 --latent_reward 0 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "hopper-medium-replay-v2" --out_name "cql-hopper-medium-replay-original"             --num_t 9000  --len_t 20 --latent_reward 0 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "hopper-medium-v2"        --out_name "cql-hopper-medium-original"                    --num_t 23000 --len_t 20 --latent_reward 0 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed

    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-expert-v2" --out_name "cql-walker2d-medium-expert-original"             --num_t 49600 --len_t 20 --latent_reward 0 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-replay-v2" --out_name "cql-walker2d-medium-replay-original"             --num_t 7000  --len_t 20 --latent_reward 0 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-v2"        --out_name "cql-walker2d-medium-original"                    --num_t 24800 --len_t 20 --latent_reward 0 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed
    
    # originals for sizes
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-original-0.5"             --num_t 49920 --len_t 20 --latent_reward 0 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0 --dataset_size_multiplier 0.5
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-original-0.2"             --num_t 49920 --len_t 20 --latent_reward 0 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0 --dataset_size_multiplier 0.2
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-original-0.1"             --num_t 49920 --len_t 20 --latent_reward 0 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0 --dataset_size_multiplier 0.1
    
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "hopper-medium-v2" --out_name "cql-hopper-medium-original-0.5"             --num_t 23000 --len_t 20 --latent_reward 0 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0 --dataset_size_multiplier 0.5
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "hopper-medium-v2" --out_name "cql-hopper-medium-original-0.2"             --num_t 23000 --len_t 20 --latent_reward 0 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0 --dataset_size_multiplier 0.2
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "hopper-medium-v2" --out_name "cql-hopper-medium-original-0.1"             --num_t 23000 --len_t 20 --latent_reward 0 --bin_label 0 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 0 --dataset_size_multiplier 0.1
done

