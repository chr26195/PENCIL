n=3
dataset="PUZ10K"

# # Run the generator
# python einstein_generator.py \
#     --num_samples 10000 \
#     --data_dir data/${dataset} \
#     --size $n \
#     --minimal_conditions \
#     --save

# # Run the solver
# python einstein_solver.py \
#     --data_dir data/${dataset} \
#     --train_size 1000

python train_puzzle.py \
    config/config_puzzle.py \
    --dataset=$dataset \
    --data_dir=data \
    --device=cuda \
    --format=pencil \
    --eval_interval=200 