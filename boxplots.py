
"""
This script visualizes three different types of distributions — normal, left-skewed, and right-skewed — using histograms and boxplots.
The goal is to understand how the **median** behaves in each case and how it compares to the **mean**.
Histograms help reveal the shape and skewness of the distributions, while boxplots summarize key statistics (median, quartiles, and outliers).
By comparing both visualizations side by side, we can see how skewness affects the position of the median and the symmetry of the box.
"""

import numpy as np
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(42)

# Create three distributions
normal = np.random.normal(loc=50, scale=10, size=1000)           # Normal distribution
right_skewed = np.random.beta(a=2, b=5, size=1000) * 100          # Right-skewed (Beta scaled)
left_skewed = np.random.beta(a=5, b=2, size=1000) * 100           # Left-skewed (Beta scaled)

# Titles for each distribution
titles = ['Normal', 'Left-Skewed', 'Right-Skewed']
data_sets = [normal, left_skewed, right_skewed]

# Create the figure with 2 rows (histograms + boxplots)
fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

# Plot histograms
for i, data in enumerate(data_sets):
    axs[0, i].hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axs[0, i].axvline(np.median(data), color='red', linestyle='--', label='Median')
    axs[0, i].axvline(np.mean(data), color='green', linestyle=':', label='Mean')
    axs[0, i].set_title(f'{titles[i]} Distribution')
    axs[0, i].legend()

# Plot boxplots
for i, data in enumerate(data_sets):
    axs[1, i].boxplot(data, vert=False)
    axs[1, i].set_title(f'Boxplot: {titles[i]}')

plt.show()