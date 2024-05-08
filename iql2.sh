#!/bin/bash

for seed in 1 2
do
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "walker2d-medium-replay-v2" --out_name "iql-walker-medium-replay-original-no_overlap"               --num_t 7000  --len_t 20 --latent_reward 0 --bin_label 0 --seed $seed
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "walker2d-medium-replay-v2" --out_name "iql-walker-medium-replay-binary_labels-no_overlap"          --num_t 7000  --len_t 20 --latent_reward 0 --bin_label 1 --seed $seed
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "walker2d-medium-replay-v2" --out_name "iql-walker-medium-replay-latent_reward-no_overlap"          --num_t 7000  --len_t 20 --latent_reward 1 --bin_label 0 --seed $seed
done

