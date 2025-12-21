"""Generate density estimation plot."""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_utils as utils

def generate():
    from sklearn.neighbors import KernelDensity
    
    np.random.seed(42)
    N = 200
    X = np.concatenate((
        np.random.normal(0, 0.8, int(N * 0.6)),
        np.random.normal(4, 1.2, int(N * 0.4))
    ))[:, np.newaxis]
    
    X_plot = np.linspace(-4, 8, 1000)[:, np.newaxis]
    
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    
    # A. High Bias Histogram
    h_wide = 2.5
    bins_wide = np.arange(X.min(), X.max() + h_wide, h_wide)
    axes[0].hist(X.ravel(), bins=bins_wide, density=True, 
                 edgecolor='white', color=utils.COLORS['primary'], alpha=0.7)
    axes[0].set_title(f'Histogram ($h={h_wide}$): Overly Smooth (High Bias)')
    
    # B. High Variance Histogram
    h_small = 0.15
    bins_small = np.arange(X.min(), X.max() + h_small, h_small)
    axes[1].hist(X.ravel(), bins=bins_small, density=True, 
                 edgecolor='none', color=utils.COLORS['cyan'], alpha=0.8)
    axes[1].set_title(f'Histogram ($h={h_small}$): Noisy (High Variance)')
    
    # C. KDE
    h_kde = 0.5
    kde = KernelDensity(kernel='gaussian', bandwidth=h_kde).fit(X)
    log_dens = kde.score_samples(X_plot)
    
    axes[2].plot(X_plot.ravel(), np.exp(log_dens), 
                 color=utils.COLORS['red'], linewidth=2, label='KDE Estimate')
    axes[2].fill_between(X_plot.ravel(), np.exp(log_dens), 
                         color=utils.COLORS['red'], alpha=0.2)
    axes[2].set_title(f'Kernel Density Estimation ($h={h_kde}$): Smooth & Flexible')
    
    for ax in axes:
        ax.plot(X.ravel(), np.full_like(X, -0.01), '|', 
                color=utils.COLORS['dark'], markeredgewidth=1, label='Data Points')
        ax.set_ylabel('Density')
    
    axes[2].set_xlabel('Feature Value ($x$)')
    axes[2].legend(loc='upper left')
    
    utils.save_figure("density_estimation.pdf")

if __name__ == "__main__":
    utils.set_style()
    generate()
