#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data

mkdir -p $data

tools=$base/tools

# Ensure the base data directory exists
mkdir -p $data/movie_dialogs

# link default training data for easier access

mkdir -p "$data/wikitext-2"

for corpus in train valid test; do
    # Using 'cd' and 'pwd' to emulate 'realpath' as macOS has no realpath by default
    if [ -f "$tools/pytorch-examples/word_language_model/data/wikitext-2/$corpus.txt" ]; then
        cd "$tools/pytorch-examples/word_language_model/data/wikitext-2"
        absolute_path="$(pwd)/$corpus.txt"
        cd - > /dev/null  # Go back to the previous directory without output
    else
        echo "Error: File not found - $tools/pytorch-examples/word_language_model/data/wikitext-2/$corpus.txt"
        continue  # Skip this iteration if the file does not exist
    fi

    ln -snf "$absolute_path" "$data/wikitext-2/$corpus.txt"
done

# download a different interesting data set!

mkdir -p $data/movie_dialogs

mkdir -p $data/movie_dialogs/raw

# The movie-dialog corpus has been manually downloaded and movie_lines.txt of the corpus was used and placed under 'raw' directory
# No need to download, just preprocess the existing data

# wget https://www.gutenberg.org/files/52521/52521-0.txt
# mv movie-dialogs/movie_lines.txt $data/movie_dialogs/raw/movie_lines.txt
cp movie-dialogs/movie_lines.txt $data/movie_dialogs/raw/movie_lines.txt

# Preprocess slightly
cat $data/movie_dialogs/raw/movie_lines.txt | python $base/scripts/preprocess_raw.py > $data/movie_dialogs/raw/cleaned_movie_lines.txt

# tokenize, fix vocabulary upper bound
cat $data/movie_dialogs/raw/cleaned_movie_lines.txt | python $base/scripts/preprocess.py --vocab-size 10000 --tokenize --lang "en" --sent-tokenize > \
    $data/movie_dialogs/raw/preprocessed_movie_lines.txt

# split into train, valid and test
total_lines=$(wc -l < $data/movie_dialogs/raw/preprocessed_movie_lines.txt)
num_train=$((total_lines * 80 / 100))
num_valid=$((total_lines * 10 / 100))

head -n $num_train $data/movie_dialogs/raw/preprocessed_movie_lines.txt > $data/movie_dialogs/train.txt
head -n $(($num_train + $num_valid)) $data/movie_dialogs/raw/preprocessed_movie_lines.txt | tail -n $num_valid > $data/movie_dialogs/valid.txt
tail -n +$(($num_train + $num_valid + 1)) $data/movie_dialogs/raw/preprocessed_movie_lines.txt > $data/movie_dialogs/test.txt
