import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the 'logs' folder directly
logs_folder = 'logs'

# Define the dropout rates based on the file names
dropout_rates = ['0', '0.1', '0.3', '0.5', '0.7', '0.9']

# Load data from the first file to determine the number of epochs
data = pd.read_csv(os.path.join(logs_folder, f'perplexity_log_dropout_{dropout_rates[0]}.csv'))
num_epochs = len(data) - 1  # Exclude the 'End of Training' row

# Plot the training perplexity for each dropout rate
for rate in dropout_rates:
    train_perplexity = []
    for epoch in range(1, num_epochs + 1):
        # Load the data for the current dropout rate
        df = pd.read_csv(os.path.join(logs_folder, f'perplexity_log_dropout_{rate}.csv'))
        train_perplexity.append(df.at[epoch - 1, 'train_perplexity'])

    # Plot the training perplexity
    plt.plot(range(1, num_epochs + 1), train_perplexity, label=f'Dropout {rate}')

# Set labels and title for the plot
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Training Perplexity')
plt.legend()
plt.grid(True)

# Save the training perplexity chart
plt.savefig(os.path.join(logs_folder, 'training_perplexity_chart.png'))
plt.show()

# Plot the validation perplexity for each dropout rate
for rate in dropout_rates:
    valid_perplexity = []
    for epoch in range(1, num_epochs + 1):
        # Load the data for the current dropout rate
        df = pd.read_csv(os.path.join(logs_folder, f'perplexity_log_dropout_{rate}.csv'))
        valid_perplexity.append(df.at[epoch - 1, 'valid_perplexity'])

    # Plot the validation perplexity
    plt.plot(range(1, num_epochs + 1), valid_perplexity, label=f'Dropout {rate}')

# Set labels and title for the plot
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Validation Perplexity')
plt.legend()
plt.grid(True)

# Save the validation perplexity chart
plt.savefig(os.path.join(logs_folder, 'validation_perplexity_chart.png'))
plt.show()
