#! /bin/bash

# Define the base directory relative to the script location
scripts=$(dirname "$0")
base=$(cd "$scripts"/.. && pwd)

# Define directories for models, data, tools, and logs
models=$base/models
data=$base/data
tools=$base/tools
logs=$base/logs

mkdir -p $models
mkdir -p $logs # Create logs directory

# Define the number of threads for operations
num_threads=4
device=""

# Array of dropout rates to train models with different settings
dropout_rates=(0 0.1 0.3 0.5 0.7 0.9)

# Loop through each dropout rate and perform training
for dropout in "${dropout_rates[@]}"; do
    echo "Training with dropout rate: $dropout"
    SECONDS=0

    # Navigate to the training script directory and execute the Python training script
    (cd $tools/pytorch-examples/word_language_model &&
        OMP_NUM_THREADS=$num_threads python main.py --data $data/movie_dialogs \
            --epochs 25 \
            --log-interval 100 \
            --emsize 250 \
            --nhid 250 \
            --dropout $dropout \
            --tied \
            --save $models/model_dropout_$dropout.pt \
            --log-file $logs/perplexity_log_dropout_$dropout.csv \
            --mps
    )

    # Output the time taken for training with the current dropout rate
    echo "Time taken for dropout $dropout: $SECONDS seconds"
done

