import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from pathlib import Path

def plot_performances_indicators(df, PER_df):
    ## Create a figure with two subplots
    fig, axs = plt.subplots(1, 3, figsize=(20, 5)) # 1 row, 3 columns
    
    # Plot for scores and average scores with a red line indicating the target score of 0.5
    axs[0].plot(df['Episode'], df['Score'], label='Score per Episode', color='blue')
    axs[0].plot(df['Episode'], df['Average Scores'], label='Average Score Over 100 Episodes', color='green')
    axs[0].axhline(y=0.5, color='red', linestyle='-', label='Target Score (0.5)')
    axs[0].set_title('Scores Over Episodes')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Score')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot for Actor losses and average Actor losses
    axs[1].plot(df['Episode'], df['Actor losses'], label='Average Actor Loss Over 100 Learning', color='orangered')
    axs[1].set_title('Actor Losses Over Learning')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Actor Loss')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot for Critic losses and average Actor losses
    axs[2].plot(df['Episode'], df['Critic losses'], label='Average Critic Loss Over 100 Learning', color='orangered')
    axs[2].set_title('Critic Losses Over Learning')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Critic Loss')
    axs[2].legend()
    axs[2].grid(True)
    
    # Adjust the layout to prevent overlapping
    plt.tight_layout()
    plt.savefig('plots/score_and_loss_plots.png') 
    plt.show()

    ## Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 4)) # 1 row, 2 columns
    
    # Plot for Noise Sigma
    axs[0].plot(df['Episode'], df['Noise Sigma'], label='Noise Sigma per Episode', color='blue')
    axs[0].set_title('Noise Sigma Over Episodes')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Score')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot for PER Beta
    axs[1].plot(df['Episode'], df['PER Beta'], label='Beta (PER) per Episode', color='blue')
    axs[1].set_title('Beta (from Prioritized Experience Replay) Over Episodes')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Beta')
    axs[1].legend()
    axs[1].grid(True)
    
    # Adjust the layout to prevent overlapping
    plt.tight_layout()
    plt.savefig('plots/noise_and_PER_plots.png') 
    plt.show()


# Plot PrioritizedReplayBuffer parameter
def plot_per_parameters(weight_magnitudes, td_errors_history):
    
    fig, axs = plt.subplots(2, 1, figsize=(6, 6))

    axs[0].plot(range(1, len(weight_magnitudes) + 1), weight_magnitudes, label='Average Weight Magnitude')
    axs[0].set_title('Evolution of Importance Sampling Weights')
    axs[0].set_xlabel('Learn')
    axs[0].set_ylabel('Average Weight Magnitude')
    axs[0].legend()

    axs[1].plot(range(1, len(td_errors_history) + 1), td_errors_history, label='Average TD Error')
    axs[1].set_title('Evolution of TD Errors')
    axs[1].set_xlabel('Learn')
    axs[1].set_ylabel('Average TD Error')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig('plots/weight_magnitudes_and_td_errors_plots.png') 
    plt.show()


# resize weigts and gradient data
def resize_data(param_df):
    # Load dataframe
    #param_df = pd.read_parquet('gradients_bias.parquet')
    
    # Initialize
    all_data = []
    
    # Itérer sur chaque enregistrement dans votre DataFrame
    for index, row in param_df.iterrows():
        # Extracts weight and gradient data for actor and critic
        for weight_data in row['actor_weights']:
            weight_data['episode'] = row['episode']
            weight_data['type'] = 'actor'
            weight_data['param_type'] = 'weight'
            all_data.append(weight_data)
    
        for grad_data in row['actor_gradients']:
            grad_data['episode'] = row['episode']
            grad_data['type'] = 'actor'
            grad_data['param_type'] = 'gradient'
            all_data.append(grad_data)
    
        for weight_data in row['critic_weights']:
            weight_data['episode'] = row['episode']
            weight_data['type'] = 'critic'
            weight_data['param_type'] = 'weight'
            all_data.append(weight_data)
    
        for grad_data in row['critic_gradients']:
            grad_data['episode'] = row['episode']
            grad_data['type'] = 'critic'
            grad_data['param_type'] = 'gradient'
            all_data.append(grad_data)
    
    combined_df = pd.DataFrame(all_data)
    
    weights_pivot = combined_df.pivot_table(index=['layer', 'episode', 'type'], values=['weight_mean', 'weight_std'], aggfunc='first').reset_index()
    grads_pivot = combined_df.pivot_table(index=['layer', 'episode', 'type'], values=['grad_mean', 'grad_std'], aggfunc='first').reset_index()
    merged_df = pd.merge(weights_pivot, grads_pivot, on=['layer', 'episode', 'type'], how='left')
    merged_df.fillna(0, inplace=True)    
    merged_df.sort_values(by='episode', ascending=True, inplace=True)
    
    # Save DataFrame
    merged_df.to_csv('data/combined_data.csv', index=False)
    
    return merged_df


# plot weigts and gradient 
def plot_weights_grad(df, actor_or_critic, title):
    # Filter data for actor or critic only
    df_filtered = df[df['type'] == actor_or_critic]

    # Assuming we have two fc layers and one bn layer as per the description
    layers = ['fc1','bn1', 'fc2', 'fc3']
    
    # Set up the plot grid
    fig, axes = plt.subplots(4, len(layers), figsize=(20, 10), sharex=True)  # Adjust figsize as needed
    fig.suptitle(title, fontsize=16)

    # Row titles
    row_titles = ['Weight.weight', 'Weight.bias', 'Gradient.weight', 'Gradient.bias'] #Bias(weight) Mean±STD
    
    # Plot each parameter type in its row
    for i, row_title in enumerate(row_titles):
        for j, layer in enumerate(layers):
            ax = axes[i, j]
            # Select the appropriate data for the current layer
            if 'Weight.' in row_title:
                data_type = 'weight'
            #else:
            if 'Gradient.' in row_title:
                data_type = 'grad'
            
            param_type = 'bias' if '.bias' in row_title else 'weight'
            layer_data = df_filtered[df_filtered['layer'].str.contains(layer) & df_filtered['layer'].str.contains(param_type)]

            # Plot the lineplot for the mean values
            sns.lineplot(ax=ax, data=layer_data, x='episode', y=f'{data_type}_mean', label='Mean', color='blue')
            
            # Plot the fill_between for the std deviation
            ax.fill_between(layer_data['episode'], layer_data[f'{data_type}_mean'] - layer_data[f'{data_type}_std'],
                            layer_data[f'{data_type}_mean'] + layer_data[f'{data_type}_std'], alpha=0.3, color='blue', label='STD')
            
            # Set titles and labels
            ax.set_title(f'{layer} - {row_title}')
            ax.set_xlabel('Episode')
            ax.set_ylabel(f'{data_type.capitalize()} {param_type.capitalize()}')
            ax.legend()

    # Adjust layout and display the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'plots/{actor_or_critic}_weights_and_grad.png') 
    plt.show()


# Save printed results to a text file
def save_filtered_df_to_txt(df, print_interval, filename="data/printed_results.txt"):
    # Filter columns
    columns_to_keep = ["Episode","Time", "Average Scores", "Noise Sigma", "PER Beta", "Average Actor losses", "Average Critic losses"]
    df_filtered = df[columns_to_keep]
    
    # Keep only lines at specified intervals plus last line
    rows_to_keep = range(print_interval - 1, len(df), print_interval)
    if (len(df) - 1) % print_interval != 0:  # Add last line if necessary
        rows_to_keep = list(rows_to_keep) + [len(df) - 1]
    df_filtered = df_filtered.iloc[rows_to_keep]
    
    # Save the filtered DataFrame in a text file
    with open(filename, 'w') as file:
        df_filtered.to_string(file, index=False)


def create_training_folder():
    # Use current directory as folder root
    root_path = '.'
    # Create the 'tests' folder if it doesn't exist
    tests_path = os.path.join(root_path, 'tests')
    os.makedirs(tests_path, exist_ok=True)

    # Find the next test number by examining existing files
    existing_tests = [d for d in os.listdir(tests_path) if os.path.isdir(os.path.join(tests_path, d)) and d.startswith('test_')]
    if existing_tests:
        highest_num = max(int(d.split('_')[1]) for d in existing_tests)
        next_test_num = highest_num + 1
    else:
        next_test_num = 1

    # Create the folder for the training session just completed
    next_test_path = os.path.join(tests_path, f'test_{next_test_num}')
    os.makedirs(next_test_path, exist_ok=True)

    # List of folders and files to copy
    items_to_copy = ["config.py", "Tennis.ipynb", "ddpg_agent_multi.py", "model.py", "data", "plots", "weights"]

    # Copy the specified folders and files to the test folder
    for item in items_to_copy:
        source_path = os.path.join(root_path, item)
        if os.path.isdir(source_path):
            shutil.copytree(source_path, os.path.join(next_test_path, os.path.basename(item)))
        elif os.path.isfile(source_path):
            shutil.copy2(source_path, next_test_path)

    print(f"New training data saved in: {next_test_path}")


def plot_test_scores(root_dir, column_name, episodes=170, target_score=30.0):
    root_path = Path(root_dir)
    dataframes = []
    test_names = []

    # Sort test directories numerically
    test_dirs = sorted(root_path.glob('test_*'), key=lambda x: int(x.name.split('_')[1]))

    for test_dir in test_dirs:
        parquet_file = test_dir / 'data' / 'results.parquet'
        if parquet_file.exists():
            df = pd.read_parquet(parquet_file)
            # Check if column_name exists in df
            if column_name in df.columns:
                # Extract the specified column and rename it
                df_renamed = df[[column_name]].rename(columns={column_name: test_dir.name})
                dataframes.append(df_renamed)
                test_names.append(test_dir.name)

    # Concatenate all dataframes along the columns (axis=1)
    final_df = pd.concat(dataframes, axis=1)

    # Plotting
    plt.figure(figsize=(8, 5))
    for col in final_df.columns:
        plt.plot(final_df.index[:episodes], final_df[col][:episodes], label=col)

    plt.axhline(y=target_score, color='red', linestyle='--', label=f'Target Score ({target_score})')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.6), ncol=3)
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.title('Scores from Different Tests')
    plt.show()