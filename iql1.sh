#!/bin/bash

for seed in 1 2
do
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "hopper-medium-replay-v2" --out_name "iql-hopper-medium-replay-original-no_overlap"                   --num_t 9000  --len_t 20 --latent_reward 0 --bin_label 0 --seed $seed --bin_label_allow_overlap 0
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "hopper-medium-replay-v2" --out_name "iql-hopper-medium-replay-binary_labels-no_overlap"              --num_t 9000  --len_t 20 --latent_reward 0 --bin_label 1 --seed $seed --bin_label_allow_overlap 0
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "hopper-medium-replay-v2" --out_name "iql-hopper-medium-replay-latent_reward-no_overlap"              --num_t 9000  --len_t 20 --latent_reward 1 --bin_label 0 --seed $seed --bin_label_allow_overlap 0
done

