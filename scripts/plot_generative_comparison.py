"""Generate generative classifier comparison plot."""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_utils as utils

def generate():
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
    from sklearn.neighbors import NearestCentroid
    
    np.random.seed(42)
    n_samples = 120
    
    mean0, cov0 = [0, 0], [[3.0, 2.0], [2.0, 2.0]]
    X0 = np.random.multivariate_normal(mean0, cov0, n_samples)
    y0 = np.zeros(n_samples)
    
    mean1, cov1 = [4, 3], [[2.5, -1.5], [-1.5, 2.5]]
    X1 = np.random.multivariate_normal(mean1, cov1, n_samples)
    y1 = np.ones(n_samples)
    
    X = np.vstack((X0, X1))
    y = np.hstack((y0, y1))
    
    classifiers = [
        (QuadraticDiscriminantAnalysis(store_covariance=True), "QDA: Unique $\\Sigma_i$"),
        (LinearDiscriminantAnalysis(store_covariance=True), "LDA: Shared $\\Sigma$"),
        (NearestCentroid(), "NMC: $\\Sigma = \\sigma^2 I$")
    ]
    
    mean_marker_style = dict(marker='o', s=120, c='white', edgecolors='black', linewidth=2, zorder=10)
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True, sharex=True)
    
    for ax, (clf, name) in zip(axes, classifiers):
        clf.fit(X, y)
        utils.plot_decision_regions(ax, clf, X, y)
        
        if hasattr(clf, 'means_'):
            means = clf.means_
        elif hasattr(clf, 'centroids_'):
            means = clf.centroids_
        ax.scatter(means[:, 0], means[:, 1], **mean_marker_style, label='Class Means')
        
        if "QDA" in name:
            utils.plot_gaussian_ellipse(ax, clf.means_[0], clf.covariance_[0], utils.COLORS['primary'])
            utils.plot_gaussian_ellipse(ax, clf.means_[1], clf.covariance_[1], utils.COLORS['red'])
        elif "LDA" in name:
            utils.plot_gaussian_ellipse(ax, clf.means_[0], clf.covariance_, utils.COLORS['primary'])
            utils.plot_gaussian_ellipse(ax, clf.means_[1], clf.covariance_, utils.COLORS['red'])
        elif "NMC" in name:
            avg_var = np.var(X)
            identity_cov = avg_var * np.eye(2)
            utils.plot_gaussian_ellipse(ax, means[0], identity_cov, utils.COLORS['primary'])
            utils.plot_gaussian_ellipse(ax, means[1], identity_cov, utils.COLORS['red'])
        
        ax.set_title(name)
        ax.set_xlabel(r'$x_1$')
    
    axes[0].set_ylabel(r'$x_2$')
    axes[0].legend(loc='lower right')
    
    utils.save_figure("generative_comparison.pdf")

if __name__ == "__main__":
    utils.set_style()
    generate()
