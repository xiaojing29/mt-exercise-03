#! /bin/bash

scripts=$(dirname "$0")
# Using 'cd' and 'pwd' to replace 'realpath' for better compatibility, especially on macOS
base=$(cd "$scripts"/.. && pwd)

models=$base/models
data=$base/data
tools=$base/tools
samples=$base/samples

mkdir -p $samples

num_threads=4
device=""


(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/movie_dialogs \
        --words 100 \
        --checkpoint $models/model_dropout_0.9.pt \
        --outf $samples/sample_dropout_0.9 \
        --mps \
        --temperature 0.6
)
