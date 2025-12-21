"""Generate FLD projection plot."""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_utils as utils

def generate():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    np.random.seed(42)
    n_samples = 100
    
    X0 = np.random.multivariate_normal([2, 3], [[1, 0.8], [0.8, 1]], n_samples)
    X1 = np.random.multivariate_normal([5, 6], [[1, -0.6], [-0.6, 1]], n_samples)
    
    X = np.vstack((X0, X1))
    y = np.hstack((np.zeros(n_samples), np.ones(n_samples)))
    
    lda = LinearDiscriminantAnalysis().fit(X, y)
    w = lda.scalings_[:, 0]
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [1, 1]})
    colors = [utils.COLORS['primary'], utils.COLORS['red']]
    
    # Panel 1: Naive Projection
    ax1 = axes[0]
    proj_bad = X[:, 0]
    
    for i, c in enumerate(colors):
        ax1.scatter(X[y == i, 0], X[y == i, 1], alpha=0.5, c=c, label=f'Class {i}')
        ax1.hist(proj_bad[y == i], bins=20, density=True, alpha=0.6, color=c, 
                 range=(0, 8), label='_nolegend_')
    
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_title(r'Naive Projection (onto $x_1$-axis)')
    ax1.set_xlabel(r'$x_1$ (Projection Axis)')
    ax1.set_ylabel(r'$x_2$')
    ax1.legend()
    
    # Panel 2: FLD Projection
    ax2 = axes[1]
    proj_good = X @ w
    
    x_vals = np.linspace(0, 8, 100)
    center = X.mean(axis=0)
    y_vals = (w[1] / w[0]) * (x_vals - center[0]) + center[1]
    ax2.plot(x_vals, y_vals, '--', color=utils.COLORS['dark'], 
             linewidth=2, label=r'FLD Axis $\mathbf{w}$')
    
    for i, c in enumerate(colors):
        ax2.scatter(X[y == i, 0], X[y == i, 1], alpha=0.5, c=c, label=f'Class {i}')
    
    # Inset Histogram
    ax_hist = ax2.inset_axes([0.6, 0.1, 0.35, 0.25])
    for i, c in enumerate(colors):
        ax_hist.hist(proj_good[y == i], bins=20, density=True, alpha=0.7, color=c)
    ax_hist.set_title('Projected Data', fontsize=8)
    ax_hist.set_yticks([])
    ax_hist.set_xticks([])
    
    ax2.set_title(r'Fisher Linear Discriminant (FLD) Projection')
    ax2.set_xlabel(r'$x_1$')
    ax2.set_ylabel(r'$x_2$')
    ax2.legend(loc='upper left')
    
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    
    utils.save_figure("fld_projection.pdf")

if __name__ == "__main__":
    utils.set_style()
    generate()
