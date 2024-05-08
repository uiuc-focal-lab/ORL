#!/bin/bash

for seed in 1 2
do
    python CORL/algorithms/offline/edac.py --project "Experiments_CORL" --env_name "walker2d-medium-v2" --out_name "edac-walker-medium-original-no_overlap"               --num_t 24000  --len_t 20 --latent_reward 0 --bin_label 0 --train_seed $seed --eval_seed $seed
    python CORL/algorithms/offline/edac.py --project "Experiments_CORL" --env_name "walker2d-medium-v2" --out_name "edac-walker-medium-binary_labels-no_overlap"          --num_t 24000  --len_t 20 --latent_reward 0 --bin_label 1 --train_seed $seed --eval_seed $seed
    python CORL/algorithms/offline/edac.py --project "Experiments_CORL" --env_name "walker2d-medium-v2" --out_name "edac-walker-medium-latent_reward-no_overlap"          --num_t 24000  --len_t 20 --latent_reward 1 --bin_label 0 --train_seed $seed --eval_seed $seed
done

