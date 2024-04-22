# MT Exercise 3: Pytorch RNN Language Models

This repo shows how to train neural language models using [Pytorch example code](https://github.com/pytorch/examples/tree/master/word_language_model). Thanks to Emma van den Bold, the original author of these scripts. 

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

## Modifications

**Changes to `preprocess_raw.py` Script for Task 1**:
- Updated `preprocess_raw.py` to enhance data handling and improve the quality of the Cornell Movie-Dialogs Corpus used for model training. Key changes include:
  - **Dialogue Parsing and Cleaning**: Enhanced the script to split each line on the delimiter "+++$+++", which separates metadata from the actual dialogue. The script now more effectively extracts the dialogue part, removes unwanted characters like Unicode BOM marks and newline characters, and normalizes whitespace for cleaner and more uniform text data.
  - **Redundancy Reduction**: Introduced a new function to identify and eliminate redundant or less informative dialogues. This function checks for short dialogues with fewer than three words and dialogues with a high proportion of repetitive words, reducing noise and focusing on more meaningful text. This preprocessing step helps in pruning the dataset to include only substantive and informative dialogues, ensuring the model trains on high-quality data.
  - **Data Quality Improvement**: These changes collectively improve the overall data quality, making it more suitable for training sophisticated neural language models by providing cleaner, more relevant, and contextually rich training examples.
  - **Data Extraction**: Took 1/5 of the dataset to train a model to speed up training.

**Changes to `download_data.sh` Script for Task 1***:
- Updated the `download_data.sh` to handle data preprocessing for the Cornell Movie-Dialogs Corpus, replacing the previous dataset:
  - Modified path handling to use `cd` and `pwd` instead of `realpath` for better compatibility with macOS.
  - Replaced the dataset download section with steps to handle movie dialogues that have been manually placed in the `raw` directory.
  - Introduced dynamic calculation for train, valid, and test splits based on the total number of lines, ensuring accurate data segmentation for model training.
  - Changed vocab-size to 10000 as we have a very large dataset with 446005 lines in total.

**Changes to `train.sh` Script for Task 1**:
- Adapted `train.sh` to accommodate the infrastructure and dataset changes:
  - Replaced `realpath` with `cd` and `pwd` for improved compatibility on macOS.
  - Changed the data directory in the script to `$data/movie_dialogs` to align with the new dataset focus.
  - Added the `--mps` flag to enable GPU support on macOS with Apple Silicon, enhancing training performance on compatible systems.

**Changes to `train.sh` Script for Task 2**:
- Logs Directory: Added a `logs` directory to store training logs for different dropout rates.
- Multiple Dropout Rates: Introduced training across a range of dropout rates (0.0, 0.1, 0.3, 0.5, 0.7, 0.9) to observe the impact on model performance.
- Epochs Reduced: Training epochs reduced from 40 to 25 to expedite training iterations.
- Increased Model Complexity: Increased embedding size and number of hidden units from 200 to 250 to potentially enhance model capability.
- Detailed Training Logs: Added logging for each dropout rate training session with specific files per training session to make performance tracking easier.
- Model Saving Convention: Models are now saved with names that reflect their respective dropout settings, e.g., `model_dropout_0.1.pt`.

**Changes to `data.py` Script for Task 1***:
- Enhanced error handling and efficiency in data processing:
  - Replaced the assertion for file existence with an exception handling mechanism to provide clearer error messages (`FileNotFoundError`).
  - Streamlined the tokenization process to occur in a single read-through of the file, reducing the computational overhead and improving the speed of data preparation.

**Changes to `generate.sh` Script for Task 1***:
- Updated the script to ensure compatibility and functionality with the trained model and data:
  - Modified directory paths to correctly point to the movie dialogues dataset and model files.
  - Replaced `realpath` with `cd` and `pwd` to maintain compatibility with macOS environments.

**Changes to `main.py` Script for Task 2**
- Added import statement for the `csv` module.
- Added a new command-line argument `--log-file` to save the log file for perplexities.
- Defined a new function `log_perplexities` to log perplexities, including training, validation, and test perplexities, to the specified CSV file.
- Modified the training loop to calculate and log the training perplexity at each epoch, as well as to log the perplexities for both validation and test datasets.
- Incorporated calls to the `log_perplexities` function within the training loop to log perplexities for each epoch, including the training, validation, and test perplexities.
- Logged the final test perplexity after completing the training using the `log_perplexities` function.
- Modified the `train` function to accept the learning rate (`lr`) as an argument.
- Updated the gradient descent step within the `train` function to use the passed learning rate (`lr`) instead of `args.lr`.
- Adjusted the learning rate (`lr`) usage in the training loop print statement to reflect the passed value instead of `args.lr`.

**Addition of `create_perplexity_tables.py` Script to Create Tables for the Three Perplexities for Task 2**
- This script generates and saves tables for three types of perplexities: validation, training, and test. Each table includes perplexity values for each epoch and dropout rate, facilitating easy comparison and analysis of model performance. 
- The script loads the perplexity logs from the 'logs' folder and creates separate CSV files for each type of perplexity. These CSV files contain tables with perplexity values across various dropout rates. 
- To run the script, simply execute it in your Python environment.

**Addition of `create_line_plots.py` Script to Visualize Training and Validation Perplexity for Task 2**
- The script is designed to visualize the training and validation perplexities across different dropout rates. It utilizes the `pandas` library to load the perplexity logs from the 'logs' folder and `matplotlib.pyplot` to generate line charts for each dropout rate.
- To utilize this script, simply execute it in your Python environment.

**Addition of `create_correlation_charts.py` Script to Visualize Connection between Training, Test and Validation Perplexity for Task 2**
- The script facilitates the comparison of training, validation, and test perplexities across different dropout rates. 
- To utilize this script, ensure that the CSV files containing perplexity data are present in the 'logs' folder and execute the script in your Python environment.


# Steps

Clone this repository in the desired place:

    git clone https://github.com/xiaojing29/mt-exercise-03
    cd mt-exercise-03

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Download and install required software:

    ./scripts/install_packages.sh

Download and preprocess data:

    ./scripts/download_data.sh

Convert data:
**Conversion from `Windows-1252` to `UTF-8`**:
- The original dataset was encoded in Windows-1252, which required conversion to UTF-8 to ensure compatibility with our processing scripts. We used the following commands to perform the conversion:


    iconv -f windows-1252 -t utf-8 data/movie_dialogs/train.txt > data/movie_dialogs/train_utf8.txt
    mv data/movie_dialogs/train_utf8.txt data/movie_dialogs/train.txt

    iconv -f windows-1252 -t utf-8 data/movie_dialogs/valid.txt > data/movie_dialogs/valid_utf8.txt
    mv data/movie_dialogs/valid_utf8.txt data/movie_dialogs/valid.txt

    iconv -f windows-1252 -t utf-8 data/movie_dialogs/test.txt > data/movie_dialogs/test_utf8.txt
    mv data/movie_dialogs/test_utf8.txt data/movie_dialogs/test.txt


Train a model:

    ./scripts/train.sh

The training process can be interrupted at any time, and the best checkpoint will always be saved.

Generate (sample) some text from a trained model with:

    ./scripts/generate.sh

Create a table each for the training, test and validation perplexity:

    python scripts/create_tables.py


Create a line chart each for the training and the validation perplexity:

    python scripts/create_line_charts.py

Create a chart to visualize the connection between training, test and validation perplexity for each dropout rate:

    python scripts/create_correlation_charts.py


