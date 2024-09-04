export DATASET_NAME=rubble
export EXP_PATH=results/rubble
export DATASET_PATH=refined_datasets/rubble
export SCRATCH_PATH=results/rubble/scratch
export MASK_PATH=refined_datasets/rubble/mask
export NUM_GPU==2
export CONFIG_FILE=configs/mega-nerf/rubble.yaml

python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node 2 mega_nerf/eval.py --config_file configs/mega-nerf/rubble.yaml  --exp_name evaluation/rubble --dataset_path refined_datasets/rubble --container_path models/rubble/rubble.pt