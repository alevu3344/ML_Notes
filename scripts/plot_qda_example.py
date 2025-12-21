"""Generate QDA example plot."""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_utils as utils

def generate():
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    
    np.random.seed(42)
    X = np.vstack([
        np.random.multivariate_normal([0, 0], [[2, 0.8], [0.8, 1]], 50),
        np.random.multivariate_normal([4, 5], [[1.5, -0.6], [-0.6, 3]], 50)
    ])
    y = np.hstack([np.zeros(50), np.ones(50)])
    
    clf = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X, y)
    
    fig, ax = plt.subplots()
    utils.plot_decision_regions(ax, clf, X, y)
    utils.plot_gaussian_ellipse(ax, clf.means_[0], clf.covariance_[0], utils.COLORS['primary'])
    utils.plot_gaussian_ellipse(ax, clf.means_[1], clf.covariance_[1], utils.COLORS['red'])
    
    utils.save_figure("qda_example.pdf")

if __name__ == "__main__":
    utils.set_style()
    generate()
