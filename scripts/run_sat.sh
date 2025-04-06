dataset="SAT100K" 

python dataset_sat.py \
    --num_samples=102000 \
    --train_size=10000 \
    --data_dir=data/${dataset} \
    --min_vars=3 \
    --max_vars=3

python train.py \
    config/config_3sat.py \
    --dataset=$dataset \
    --data_dir=data \
    --device=cuda \
    --format=pencil \