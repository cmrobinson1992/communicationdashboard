import pandas as pd
import matplotlib.pyplot as plt
import sys, os

# Assuming results_df is already available and looks something like this:
# results_df = pd.DataFrame(gridsearch.cv_results_)

# Set up the figure with 3 subplots side by side
script_dir = os.path.dirname(os.path.abspath(__file__))
results_df = pd.read_csv(os.path.join(script_dir, "./cnn_glove_cv_results.csv"))

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

activations = ['ELU', 'ReLu', 'Tanh']
learning_rates = [0.001, 0.005]
avg_df = (
    results_df
    .groupby(['param_activation', 'param_lr', 'param_max_epochs'])['mean_test_score']
    .mean()
    .reset_index()
)

colors = {0.001: '#418FDE', 0.005: '#C8102E'}
for i, activation in enumerate(activations):
    ax = axes[i]
    for lr in learning_rates:
        subset = avg_df[
            (avg_df['param_activation'] == activation) &
            (avg_df['param_lr'] == lr)
        ].sort_values(by='param_max_epochs')
        
        ax.plot(
            subset['param_max_epochs'],
            subset['mean_test_score'],
            marker='o',
            label=f'lr={lr}',
            color = colors[lr],
            linewidth = 6
        )
    
    ax.set_title(f'Activation: {activation}',fontsize=32, fontfamily='century gothic')
    ax.set_xlabel('Max Epochs', fontsize=24, fontfamily='century gothic')
    if i == 0:
        ax.set_ylabel('Avg Mean Accuracy', fontsize=24, fontfamily='century gothic')
    ax.tick_params(axis = 'both', labelsize = 20, labelfontfamily = 'century gothic')
    ax.legend()
    ax.grid(False)


plt.tight_layout()
plt.show()
