import pandas as pd
import os
import csv

# Define the path to the 'logs' folder directly, since we are in the 'mt-exercise-03' directory
logs_folder = 'logs'

# Define the dropout rates based on the file names
dropout_rates = ['0', '0.1', '0.3', '0.5', '0.7', '0.9']

# Function to create and save tables for each perplexity type
def create_and_save_tables(perplexity_types, dropout_rates, logs_folder):
    for p_type in perplexity_types:
        # Initialize the table with the header
        header = [f'{p_type.replace("_", " ").capitalize()}'] + \
                 [f'Dropout {rate}' for rate in dropout_rates]
        if p_type == 'test_perplexity':
            header[0] = header[0].replace('Perplexity', 'Perplexity'.capitalize())
        table = [header]

        # Load data from the first file to determine the number of epochs
        data = pd.read_csv(os.path.join(logs_folder, f'perplexity_log_dropout_{dropout_rates[0]}.csv'))
        num_epochs = len(data) - 1  # Exclude the 'End of Training' row

        # Collect the data for each epoch across all dropout rates
        for epoch in range(1, num_epochs + 1):
            row = [f'Epoch {epoch}']
            for rate in dropout_rates:
                # Load the data for the current dropout rate
                df = pd.read_csv(os.path.join(logs_folder, f'perplexity_log_dropout_{rate}.csv'))
                # Append the specific perplexity type value
                if p_type == 'test_perplexity':
                    if epoch == num_epochs:
                        row.append(df.at[num_epochs, p_type])
                    else:
                        row.append('')
                else:
                    row.append(df.at[epoch - 1, p_type])
            table.append(row)

        # Save the table to a CSV file
        output_file_name = f'{p_type}_table.csv'
        output_file_path = os.path.join(logs_folder, output_file_name)
        with open(output_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(table)

        print(f'{p_type.capitalize()} table saved to {output_file_path}')

# The perplexity types based on your required format
perplexity_types = ['valid_perplexity', 'train_perplexity', 'test_perplexity']

# Execute the function to create and save the tables
create_and_save_tables(perplexity_types, dropout_rates, logs_folder)
