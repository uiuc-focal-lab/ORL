for seed in 15 25 35
do
# sizes 
    # cql halfcheetah ME
    python algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-original-0.5"               --num_t 49920  --len_t 20 --latent_reward 0 --bin_label 0 --seed $seed --bin_label_allow_overlap 1 --dataset_size_multiplier 0.5
    python algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-original-0.2"               --num_t 49920  --len_t 20 --latent_reward 0 --bin_label 0 --seed $seed --bin_label_allow_overlap 1 --dataset_size_multiplier 0.2
    python algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-original-0.1"               --num_t 49920  --len_t 20 --latent_reward 0 --bin_label 0 --seed $seed --bin_label_allow_overlap 1 --dataset_size_multiplier 0.1

    python algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-binary_labels-0.5"          --num_t 49920  --len_t 20 --latent_reward 0 --bin_label 1 --seed $seed --bin_label_allow_overlap 1 --dataset_size_multiplier 0.5
    python algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-binary_labels-0.2"          --num_t 49920  --len_t 20 --latent_reward 0 --bin_label 1 --seed $seed --bin_label_allow_overlap 1 --dataset_size_multiplier 0.2
    python algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-binary_labels-0.1"          --num_t 49920  --len_t 20 --latent_reward 0 --bin_label 1 --seed $seed --bin_label_allow_overlap 1 --dataset_size_multiplier 0.1

    python algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-latent_reward-0.5"          --num_t 49920  --len_t 20 --latent_reward 1 --bin_label 0 --seed $seed --bin_label_allow_overlap 1 --dataset_size_multiplier 0.5
    python algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-latent_reward-0.2"          --num_t 49920  --len_t 20 --latent_reward 1 --bin_label 0 --seed $seed --bin_label_allow_overlap 1 --dataset_size_multiplier 0.2
    python algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-latent_reward-0.1"          --num_t 49920  --len_t 20 --latent_reward 1 --bin_label 0 --seed $seed --bin_label_allow_overlap 1 --dataset_size_multiplier 0.1

    # iql hopper MR
    python algorithms/offline/iql.py --project "Experiments_CORL" --env "hopper-medium-replay-v2" --out_name "iql-hopper-medium-replay-original-no_overlap-0.5"               --num_t 9000  --len_t 20 --latent_reward 0 --bin_label 0 --seed $seed --bin_label_allow_overlap 0 --dataset_size_multiplier 0.5
    python algorithms/offline/iql.py --project "Experiments_CORL" --env "hopper-medium-replay-v2" --out_name "iql-hopper-medium-replay-original-no_overlap-0.2"               --num_t 9000  --len_t 20 --latent_reward 0 --bin_label 0 --seed $seed --bin_label_allow_overlap 0 --dataset_size_multiplier 0.2
    python algorithms/offline/iql.py --project "Experiments_CORL" --env "hopper-medium-replay-v2" --out_name "iql-hopper-medium-replay-original-no_overlap-0.1"               --num_t 9000  --len_t 20 --latent_reward 0 --bin_label 0 --seed $seed --bin_label_allow_overlap 0 --dataset_size_multiplier 0.1

    python algorithms/offline/iql.py --project "Experiments_CORL" --env "hopper-medium-replay-v2" --out_name "iql-hopper-medium-replay-binary_labels-no_overlap-0.5"          --num_t 9000  --len_t 20 --latent_reward 0 --bin_label 1 --seed $seed --bin_label_allow_overlap 0 --dataset_size_multiplier 0.5
    python algorithms/offline/iql.py --project "Experiments_CORL" --env "hopper-medium-replay-v2" --out_name "iql-hopper-medium-replay-binary_labels-no_overlap-0.2"          --num_t 9000  --len_t 20 --latent_reward 0 --bin_label 1 --seed $seed --bin_label_allow_overlap 0 --dataset_size_multiplier 0.2
    python algorithms/offline/iql.py --project "Experiments_CORL" --env "hopper-medium-replay-v2" --out_name "iql-hopper-medium-replay-binary_labels-no_overlap-0.1"          --num_t 9000  --len_t 20 --latent_reward 0 --bin_label 1 --seed $seed --bin_label_allow_overlap 0 --dataset_size_multiplier 0.1

    python algorithms/offline/iql.py --project "Experiments_CORL" --env "hopper-medium-replay-v2" --out_name "iql-hopper-medium-replay-latent_reward-no_overlap-0.5"          --num_t 9000  --len_t 20 --latent_reward 1 --bin_label 0 --seed $seed --bin_label_allow_overlap 0 --dataset_size_multiplier 0.5
    python algorithms/offline/iql.py --project "Experiments_CORL" --env "hopper-medium-replay-v2" --out_name "iql-hopper-medium-replay-latent_reward-no_overlap-0.2"          --num_t 9000  --len_t 20 --latent_reward 1 --bin_label 0 --seed $seed --bin_label_allow_overlap 0 --dataset_size_multiplier 0.2
    python algorithms/offline/iql.py --project "Experiments_CORL" --env "hopper-medium-replay-v2" --out_name "iql-hopper-medium-replay-latent_reward-no_overlap-0.1"          --num_t 9000  --len_t 20 --latent_reward 1 --bin_label 0 --seed $seed --bin_label_allow_overlap 0 --dataset_size_multiplier 0.1

    # # ipl halfcheetah ME
    # python scripts/train.py --config output_configs/config_halfcheetah-medium-expert-v2-0.5.yaml --path "saved/halfcheetah-medium-expert-0.5_$seed"
    # python scripts/train.py --config output_configs/config_halfcheetah-medium-expert-v2-0.2.yaml --path "saved/halfcheetah-medium-expert-0.2_$seed"
    # python scripts/train.py --config output_configs/config_halfcheetah-medium-expert-v2-0.1.yaml --path "saved/halfcheetah-medium-expert-0.1_$seed"

    # # ipl hopper MR
    # python scripts/train.py --config output_configs/config_hopper-medium-replay-v2-0.5.yaml --path "saved/hopper-medium-replay-0.5_$seed"
    # python scripts/train.py --config output_configs/config_hopper-medium-replay-v2-0.2.yaml --path "saved/hopper-medium-replay-0.2_$seed"
    # python scripts/train.py --config output_configs/config_hopper-medium-replay-v2-0.1.yaml --path "saved/hopper-medium-replay-0.1_$seed"

# other algos
    # my_offlinerl_kit
    # # combo halfcheetah ME
    # python run_example/run_combo.py --task "halfcheetah-medium-expert-v2"        --log_dir_name "original_$seed"                       --num_t 49920 --len_t 20  --latent_reward 0 --bin_label 0 --seed $seed --rollout-length 5 --cql-weight 5.0 --use_original_dataset 0
    # python run_example/run_combo.py --task "halfcheetah-medium-expert-v2"        --log_dir_name "binary_labels_$seed"                  --num_t 49920 --len_t 20  --latent_reward 0 --bin_label 1 --seed $seed --rollout-length 5 --cql-weight 5.0 --use_original_dataset 0
    # python run_example/run_combo.py --task "halfcheetah-medium-expert-v2"        --log_dir_name "latent_reward_$seed"                  --num_t 49920 --len_t 20  --latent_reward 1 --bin_label 0 --seed $seed --rollout-length 5 --cql-weight 5.0 --use_original_dataset 0
    
    # iql halfcheetah ME
    python algorithms/offline/iql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "iql-halfcheetah-medium-expert-original"               --num_t 49920  --len_t 20 --latent_reward 0 --bin_label 0 --seed $seed --bin_label_allow_overlap 1
    python algorithms/offline/iql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "iql-halfcheetah-medium-expert-binary_labels"          --num_t 49920  --len_t 20 --latent_reward 0 --bin_label 1 --seed $seed --bin_label_allow_overlap 1
    python algorithms/offline/iql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "iql-halfcheetah-medium-expert-latent_reward"          --num_t 49920  --len_t 20 --latent_reward 1 --bin_label 0 --seed $seed --bin_label_allow_overlap 1

# multiple labels
    # cql halfcheetah ME
    python algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-binary_labels-multi_berno"          --num_t 49920  --len_t 20 --latent_reward 0 --bin_label 1 --seed $seed --bin_label_allow_overlap 1 --num_berno 10
    python algorithms/offline/cql.py --project "Experiments_CORL" --env "halfcheetah-medium-expert-v2" --out_name "cql-halfcheetah-medium-expert-latent_reward-multi_berno"          --num_t 49920  --len_t 20 --latent_reward 1 --bin_label 0 --seed $seed --bin_label_allow_overlap 1 --num_berno 10

    # iql hopper MR
    python algorithms/offline/iql.py --project "Experiments_CORL" --env "hopper-medium-replay-v2" --out_name "iql-hopper-medium-replay-binary_labels-no_overlap-multi_berno"          --num_t 9000  --len_t 20 --latent_reward 0 --bin_label 1 --seed $seed --bin_label_allow_overlap 0 --num_berno 10
    python algorithms/offline/iql.py --project "Experiments_CORL" --env "hopper-medium-replay-v2" --out_name "iql-hopper-medium-replay-latent_reward-no_overlap-multi_berno"          --num_t 9000  --len_t 20 --latent_reward 1 --bin_label 0 --seed $seed --bin_label_allow_overlap 0 --num_berno 10
done