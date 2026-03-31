import matplotlib.pyplot as plt
import numpy as np
from cycler import cycle 

def format_plot(ax, title=None, xlabel=None, ylabel=None, show_grid=True):
    """
    Applies a uniform style to a Matplotlib Axes object.
    """
    # 1. Title and Labels
    if title:
        ax.set_title(title, fontsize=14, fontweight='semibold', pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, labelpad=10)

    # 2. Spines (The 'box' around the plot)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 3. Grid settings
    if show_grid:
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 4. Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10)

    ax.legend()


    ax.figure.tight_layout()
    
    return ax

def format_multi_plot(axes, titles=None, xlabel=None, ylabel=None):
    """
    Applies uniform formatting to a single Axe or an array of Axes.
    """
    # Ensure axes is iterable even if it's a single subplot
    axes_list = np.array(axes).flatten()
    
    for i, ax in enumerate(axes_list):
        # 1. Apply individual titles if a list is provided
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=12, fontweight='semibold')
        
        # 2. General styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.grid(axis='y', linestyle=':', alpha=0.6)

        # 3. Handle shared labels (optional logic)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)

    # Automatically fix overlapping labels/titles
    plt.tight_layout()


if __name__=="__main__":
    # single plot
    # Sample Data
    x = [1, 2, 3, 4, 5]
    y = [10, 24, 33, 41, 55]

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker='o', color='#1f77b4', linewidth=2)

    # Apply the uniform format
    format_plot(
        ax, 
        title="Monthly Growth Trends", 
        xlabel="Month Index", 
        ylabel="Revenue ($)"
    )

    plt.show()

    # multi-plot
    # Data
    x = np.linspace(0, 10, 100)
    data = [np.sin(x), np.cos(x), np.tan(x), np.exp(x/10)]
    titles = ["Sine Wave", "Cosine Wave", "Tangent Wave", "Exponential"]

    # Create a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Plotting loop
    for i, ax in enumerate(axs.flatten()):
        ax.plot(x, data[i], color='teal')

    # Apply formatting to the entire grid at once
    format_multi_plot(axs, titles=titles, xlabel="Time (s)", ylabel="Amplitude")

    plt.show()