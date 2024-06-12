#!/bin/bash

seed=0

# walker

# single bernoulli
python algorithms/offline/cql.py --env "walker2d-medium-expert-v2" --out_name "walker-medium-expert"             --num_t 49600 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 1
python algorithms/offline/cql.py --env "walker2d-medium-replay-v2" --out_name "walker-medium-replay"             --num_t 7000  --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 1
python algorithms/offline/cql.py --env "walker2d-medium-v2"        --out_name "walker-medium"                    --num_t 24000 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 1

# multiple bernoulli
python algorithms/offline/cql.py --env "walker2d-medium-expert-v2" --out_name "walker-medium-expert" --num_t 49600 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 10 --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 1
python algorithms/offline/cql.py --env "walker2d-medium-replay-v2" --out_name "walker-medium-replay" --num_t 7000  --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 10 --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 1
python algorithms/offline/cql.py --env "walker2d-medium-v2"        --out_name "walker-medium"        --num_t 24000 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 10 --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 1

# hopper

# single bernoulli
python algorithms/offline/cql.py --env "hopper-medium-expert-v2" --out_name "hopper-medium-expert" --num_t 48800 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 1
python algorithms/offline/cql.py --env "hopper-medium-replay-v2" --out_name "hopper-medium-replay" --num_t 9000  --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 1
python algorithms/offline/cql.py --env "hopper-medium-v2"        --out_name "hopper-medium"        --num_t 23000 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 1

# multiple bernoulli
python algorithms/offline/cql.py --env "hopper-medium-expert-v2" --out_name "hopper-medium-expert" --num_t 48800 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 10 --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 1
python algorithms/offline/cql.py --env "hopper-medium-replay-v2" --out_name "hopper-medium-replay" --num_t 9000  --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 10 --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 1
python algorithms/offline/cql.py --env "hopper-medium-v2"        --out_name "hopper-medium"        --num_t 23000 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 10 --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 1

# halfcheetah

# single bernoulli
python algorithms/offline/cql.py --env "halfcheetah-medium-expert-v2" --out_name "halfcheetah-medium-expert"             --num_t 49920 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 1
python algorithms/offline/cql.py --env "halfcheetah-medium-replay-v2" --out_name "halfcheetah-medium-replay"             --num_t 5000  --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 1
python algorithms/offline/cql.py --env "halfcheetah-medium-v2"        --out_name "halfcheetah-medium"                    --num_t 24960 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 1  --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 1

# multiple bernoulli
python algorithms/offline/cql.py --env "halfcheetah-medium-expert-v2" --out_name "halfcheetah-medium-expert"             --num_t 49920 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 10 --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 1
python algorithms/offline/cql.py --env "halfcheetah-medium-replay-v2" --out_name "halfcheetah-medium-replay"             --num_t 5000  --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 10 --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 1
python algorithms/offline/cql.py --env "halfcheetah-medium-v2"        --out_name "halfcheetah-medium"                    --num_t 24960 --len_t 20 --latent_reward 0 --bin_label 1 --num_berno 10 --bin_label_trajectory_batch 0 --bin_label_allow_overlap 1 --seed $seed --quick_stop 1
