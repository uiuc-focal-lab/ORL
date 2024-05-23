for seed in 15 25 35
do
# my_offlinerl_kit
    # combo halfcheetah M
    python run_example/run_combo.py --task "halfcheetah-medium-v2"        --log_dir_name "originalc"                       --num_t 24960 --len_t 20  --latent_reward 0 --bin_label 0 --seed $seed --rollout-length 5 --cql-weight 0.5 --use_original_dataset 0 --bin_label_allow_overlap 1
    python run_example/run_combo.py --task "halfcheetah-medium-v2"        --log_dir_name "binary_labels_$seed"                  --num_t 24960 --len_t 20  --latent_reward 0 --bin_label 1 --seed $seed --rollout-length 5 --cql-weight 0.5 --use_original_dataset 0 --bin_label_allow_overlap 1
    python run_example/run_combo.py --task "halfcheetah-medium-v2"        --log_dir_name "latent_reward_$seed"                  --num_t 24960 --len_t 20  --latent_reward 1 --bin_label 0 --seed $seed --rollout-length 5 --cql-weight 0.5 --use_original_dataset 0 --bin_label_allow_overlap 1

    # mopo halfcheetah MR
    python run_example/run_mopo.py --task "halfcheetah-medium-replay-v2"        --log_dir_name "original_$seed"                 --num_t 5000  --len_t 20  --latent_reward 0 --bin_label 0 --seed $seed --rollout-length 5 --penalty-coef 0.5 --use_original_dataset 0 --bin_label_allow_overlap 1
    python run_example/run_mopo.py --task "halfcheetah-medium-replay-v2"        --log_dir_name "latent_reward_$seed"            --num_t 5000  --len_t 20  --latent_reward 1 --bin_label 0 --seed $seed --rollout-length 5 --penalty-coef 0.5 --use_original_dataset 0 --bin_label_allow_overlap 1
    python run_example/run_mopo.py --task "halfcheetah-medium-replay-v2"        --log_dir_name "binary_labels_$seed"            --num_t 5000  --len_t 20  --latent_reward 0 --bin_label 1 --seed $seed --rollout-length 5 --penalty-coef 0.5 --use_original_dataset 0 --bin_label_allow_overlap 1

# CORL
    # cql halfcheetah ME
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-original"               --num_t 49920  --len_t 20 --latent_reward 0 --bin_label 0 --seed $seed --bin_label_allow_overlap 1
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-binary_labels"          --num_t 49920  --len_t 20 --latent_reward 0 --bin_label 1 --seed $seed --bin_label_allow_overlap 1
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-latent_reward"          --num_t 49920  --len_t 20 --latent_reward 1 --bin_label 0 --seed $seed --bin_label_allow_overlap 1

    # iql hopper M
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "hopper-medium-v2" --out_name "iql-hopper-medium-original-no_overlap"               --num_t 23000  --len_t 20 --latent_reward 0 --bin_label 0 --seed $seed --bin_label_allow_overlap 0
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "hopper-medium-v2" --out_name "iql-hopper-medium-binary_labels-no_overlap"          --num_t 23000  --len_t 20 --latent_reward 0 --bin_label 1 --seed $seed --bin_label_allow_overlap 0
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "hopper-medium-v2" --out_name "iql-hopper-medium-latent_reward-no_overlap"          --num_t 23000  --len_t 20 --latent_reward 1 --bin_label 0 --seed $seed --bin_label_allow_overlap 0

    # iql hopper MR
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "hopper-medium-replay-v2" --out_name "iql-hopper-medium-replay-original-no_overlap"               --num_t 9000  --len_t 20 --latent_reward 0 --bin_label 0 --seed $seed --bin_label_allow_overlap 0
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "hopper-medium-replay-v2" --out_name "iql-hopper-medium-replay-binary_labels-no_overlap"          --num_t 9000  --len_t 20 --latent_reward 0 --bin_label 1 --seed $seed --bin_label_allow_overlap 0
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "hopper-medium-replay-v2" --out_name "iql-hopper-medium-replay-latent_reward-no_overlap"          --num_t 9000  --len_t 20 --latent_reward 1 --bin_label 0 --seed $seed --bin_label_allow_overlap 0

    # cql hopper ME
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "hopper-medium-expert-v2" --out_name "cql-hopper-medium-expert-original-no_overlap"               --num_t 48800  --len_t 20 --latent_reward 0 --bin_label 0 --seed $seed --bin_label_allow_overlap 0
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "hopper-medium-expert-v2" --out_name "cql-hopper-medium-expert-binary_labels-no_overlap"          --num_t 48800  --len_t 20 --latent_reward 0 --bin_label 1 --seed $seed --bin_label_allow_overlap 0
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "hopper-medium-expert-v2" --out_name "cql-hopper-medium-expert-latent_reward-no_overlap"          --num_t 48800  --len_t 20 --latent_reward 1 --bin_label 0 --seed $seed --bin_label_allow_overlap 0

    # iql walker M
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "walker2d-medium-v2" --out_name "iql-walker-medium-original-no_overlap"               --num_t 24000  --len_t 20 --latent_reward 0 --bin_label 0 --seed $seed --bin_label_allow_overlap 0
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "walker2d-medium-v2" --out_name "iql-walker-medium-binary_labels-no_overlap"          --num_t 24000  --len_t 20 --latent_reward 0 --bin_label 1 --seed $seed --bin_label_allow_overlap 0
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "walker2d-medium-v2" --out_name "iql-walker-medium-latent_reward-no_overlap"          --num_t 24000  --len_t 20 --latent_reward 1 --bin_label 0 --seed $seed --bin_label_allow_overlap 0

    # iql walker MR
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "walker2d-medium-replay-v2" --out_name "iql-walker-medium-replay-original-no_overlap"               --num_t 7000  --len_t 20 --latent_reward 0 --bin_label 0 --seed $seed --bin_label_allow_overlap 0
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "walker2d-medium-replay-v2" --out_name "iql-walker-medium-replay-binary_labels-no_overlap"          --num_t 7000  --len_t 20 --latent_reward 0 --bin_label 1 --seed $seed --bin_label_allow_overlap 0
    python CORL/algorithms/offline/iql.py --project "Experiments_CORL" --env "walker2d-medium-replay-v2" --out_name "iql-walker-medium-replay-latent_reward-no_overlap"          --num_t 7000  --len_t 20 --latent_reward 1 --bin_label 0 --seed $seed --bin_label_allow_overlap 0

    # cql walker ME
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-expert-v2" --out_name "cql-walker-medium-expert-original-no_overlap"               --num_t 49600  --len_t 20 --latent_reward 0 --bin_label 0 --seed $seed --bin_label_allow_overlap 0
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-expert-v2" --out_name "cql-walker-medium-expert-binary_labels-no_overlap"          --num_t 49600  --len_t 20 --latent_reward 0 --bin_label 1 --seed $seed --bin_label_allow_overlap 0
    python CORL/algorithms/offline/cql.py --project "Experiments_CORL" --env "walker2d-medium-expert-v2" --out_name "cql-walker-medium-expert-latent_reward-no_overlap"          --num_t 49600  --len_t 20 --latent_reward 1 --bin_label 0 --seed $seed --bin_label_allow_overlap 0

# ipl
    # ipl halfcheetah M
    python scripts/train.py --config output_configs/config_halfcheetah-medium-v2.yaml --path "saved/halfcheetah-medium_$seed"

    # ipl halfcheetah MR
    python scripts/train.py --config output_configs/config_halfcheetah-medium-replay-v2.yaml --path "saved/halfcheetah-medium-replay_$seed"

    # ipl halfcheetah ME
    python scripts/train.py --config output_configs/config_halfcheetah-medium-expert-v2.yaml --path "saved/halfcheetah-medium-expert_$seed"

    # ipl hopper M
    python scripts/train.py --config output_configs/config_hopper-medium-v2.yaml --path "saved/hopper-medium_$seed"

    # ipl hopper MR
    python scripts/train.py --config output_configs/config_hopper-medium-replay-v2.yaml --path "saved/hopper-medium-replay_$seed"

    # ipl hopper ME
    python scripts/train.py --config output_configs/config_hopper-medium-expert-v2.yaml --path "saved/hopper-medium-expert_$seed"

    # ipl walker M
    python scripts/train.py --config output_configs/config_walker2d-medium-v2.yaml --path "saved/walker2d-medium_$seed"

    # ipl walker MR
    python scripts/train.py --config output_configs/config_walker2d-medium-replay-v2.yaml --path "saved/walker2d-medium-replay_$seed"

    # ipl walker ME
    python scripts/train.py --config output_configs/config_walker2d-medium-expert-v2.yaml --path "saved/walker2d-medium-expert_$seed"
done