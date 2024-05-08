#!/bin/bash

for seed in 1 2
do
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-expert-v2" --out_name "cql-walker-medium-expert-original-no_overlap"                   --num_t 49600  --len_t 20 --latent_reward 0 --bin_label 0 --seed $seed
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-expert-v2" --out_name "cql-walker-medium-expert-binary_labels-no_overlap"              --num_t 49600  --len_t 20 --latent_reward 0 --bin_label 1 --seed $seed
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-expert-v2" --out_name "cql-walker-medium-expert-latent_reward-no_overlap"              --num_t 49600  --len_t 20 --latent_reward 1 --bin_label 0 --seed $seed
done
