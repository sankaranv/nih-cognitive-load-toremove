import matplotlib.pyplot as plt
import numpy as np

# Create random data
np.random.seed(42)  # To get consistent results
data = [np.random.rand(20) for _ in range(8)]

# Colors to use for the scatterplots
colors = ['red', 'blue', 'green']

# Create a 2x4 grid of scatterplots
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))

# Loop through each axis and plot the scatterplot with different colors
for i, ax in enumerate(axes.flatten()):
    ax.scatter(data[i], data[i + 4], color=colors[i % len(colors)], label=f'Plot {i + 1}')
    ax.set_title(f'Scatterplot {i + 1}')
    ax.set_xlabel('X values')
    ax.set_ylabel('Y values')

# Add a legend to the right side of the plot
fig.legend(loc='center right', bbox_to_anchor=(1.15, 0.5), title='Legends')

# Adjust spacing between plots
plt.tight_layout()

# Show the plot
plt.show()
