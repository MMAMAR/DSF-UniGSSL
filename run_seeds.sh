#!/bin/bash

# Create a folder for logs (if it doesn't exist)
mkdir -p logs

# Loop through 20 random seeds
for seed in $(seq 1 20)
do
  echo "Running experiment with seed $seed..."

  python eval_filter_level.py \
    --dataset=cora \
    --pretraining_type=node \
    --finetuning_type=prox \
    --filtering_threshold=14.0 \
    --pca_n_comp=35 \
    --device_number=0 \
    --seed=$seed \
    > logs/seed_${seed}.log 2>&1

  echo "✅ Finished seed $seed. Log saved to logs/seed_${seed}.log"
done
