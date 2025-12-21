"""Generate k-NN boundaries plot."""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_utils as utils

def generate():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    
    np.random.seed(42)
    n_samples = 150
    
    X = np.vstack([
        np.random.multivariate_normal([0, 0], [[2, 0.8], [0.8, 1]], n_samples),
        np.random.multivariate_normal([4, 5], [[1.5, -0.6], [-0.6, 3]], n_samples),
        np.random.multivariate_normal([0, 6], [[0.5, 0], [0, 0.5]], n_samples)
    ])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples), 2*np.ones(n_samples)])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = [
        (1, r"k-NN ($k=1$) - High Variance"),
        (15, r"k-NN ($k=15$) - High Bias")
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)
    
    for ax, (k, title) in zip(axes, models):
        clf = KNeighborsClassifier(n_neighbors=k).fit(X_scaled, y)
        utils.plot_decision_regions(ax, clf, X_scaled, y)
        ax.set_title(title)
        ax.set_xlabel(r'Scaled $x_1$')
    
    axes[0].set_ylabel(r'Scaled $x_2$')
    
    utils.save_figure("knn_boundaries.pdf")

if __name__ == "__main__":
    utils.set_style()
    generate()
