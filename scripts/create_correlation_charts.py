import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the 'logs' folder
logs_folder = 'logs'

# Load data for training, validation, and test perplexity
train_data = pd.read_csv(os.path.join(logs_folder, 'train_perplexity_table.csv'))
valid_data = pd.read_csv(os.path.join(logs_folder, 'valid_perplexity_table.csv'))
test_data = pd.read_csv(os.path.join(logs_folder, 'test_perplexity_table.csv'))

# Plot the connection between training, validation, and test perplexity for each dropout rate
dropout_rates = ['Dropout 0', 'Dropout 0.1', 'Dropout 0.3', 'Dropout 0.5', 'Dropout 0.7', 'Dropout 0.9']

for rate in dropout_rates:
    plt.plot(train_data.index, train_data[rate], label=f'Training - {rate}', marker='o')
    plt.plot(valid_data.index, valid_data[rate], label=f'Validation - {rate}', marker='o')
    plt.plot(test_data.index, test_data[rate], label=f'Test - {rate}', marker='o')

    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title(f'Perplexity for Dropout Rate {rate}')
    plt.legend()
    plt.grid(True)

    # Save the chart in the 'logs' folder
    chart_file_name = f'perplexity_chart_{rate.replace(" ", "_")}.png'
    chart_file_path = os.path.join(logs_folder, chart_file_name)
    plt.savefig(chart_file_path)
    plt.close()  # Close the plot to release memory

    print(f'Chart saved as {chart_file_path}')