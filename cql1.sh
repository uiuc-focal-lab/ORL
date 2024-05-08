#!/bin/bash

for seed in 1 2
do
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "hopper-medium-expert-v2" --out_name "cql-hopper-medium-expert-original-no_overlap"                   --num_t 48800  --len_t 20 --latent_reward 0 --bin_label 0 --seed $seed --bin_label_allow_overlap 0
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "hopper-medium-expert-v2" --out_name "cql-hopper-medium-expert-binary_labels-no_overlap"              --num_t 48800  --len_t 20 --latent_reward 0 --bin_label 1 --seed $seed --bin_label_allow_overlap 0
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "hopper-medium-expert-v2" --out_name "cql-hopper-medium-expert-latent_reward-no_overlap"              --num_t 48800  --len_t 20 --latent_reward 1 --bin_label 0 --seed $seed --bin_label_allow_overlap 0
done

