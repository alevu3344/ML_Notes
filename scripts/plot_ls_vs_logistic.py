"""Generate LS vs Logistic regression plot."""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_utils as utils

def generate():
    from sklearn.linear_model import LinearRegression, LogisticRegression
    
    np.random.seed(1)
    n = 50
    
    X0 = np.random.multivariate_normal([1, 2], [[1, 0.5], [0.5, 1]], n)
    X1 = np.random.multivariate_normal([5, 6], [[1, 0.5], [0.5, 1]], n)
    X_clean = np.vstack((X0, X1))
    y_clean = np.hstack((np.zeros(n), np.ones(n)))
    
    X_outlier = np.array([[10, 4]])
    y_outlier = np.array([0])
    X_noisy = np.vstack((X_clean, X_outlier))
    y_noisy = np.hstack((y_clean, y_outlier))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    datasets = [
        ("Clean Dataset", X_clean, y_clean), 
        ("With Outlier", X_noisy, y_noisy)
    ]
    
    x_min, x_max = -2, 12
    y_min, y_max = -2, 10
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    for ax, (title, X, y) in zip(axes, datasets):
        ls = LinearRegression().fit(X, y)
        lr = LogisticRegression().fit(X, y)
        
        ax.scatter(X[y==0, 0], X[y==0, 1], c=utils.COLORS['primary'], 
                   edgecolor='k', s=40, label='Class 0')
        ax.scatter(X[y==1, 0], X[y==1, 1], c=utils.COLORS['red'], 
                   edgecolor='k', s=40, label='Class 1')

        if "Outlier" in title:
            ax.scatter(X_outlier[:, 0], X_outlier[:, 1], marker='X', s=150,
                       c=utils.COLORS['primary'], edgecolors='k', label='Outlier', zorder=10)

        Z_ls = ls.predict(grid).reshape(xx.shape)
        ax.contour(xx, yy, Z_ls, levels=[0.5], colors=utils.COLORS['primary'], 
                   linestyles='--', linewidths=2)
        
        Z_lr = lr.predict_proba(grid)[:, 1].reshape(xx.shape)
        ax.contour(xx, yy, Z_lr, levels=[0.5], colors=utils.COLORS['red'], 
                   linestyles='-', linewidths=2)
        
        ax.plot([], [], c=utils.COLORS['primary'], ls='--', lw=2, label='Least Squares')
        ax.plot([], [], c=utils.COLORS['red'], ls='-', lw=2, label='Logistic Regression')

        ax.set_title(title)
        ax.set_xlabel(r'$x_1$')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle=':', alpha=0.6)
    
    axes[0].set_ylabel(r'$x_2$')
    axes[1].legend(loc='upper left')
    
    utils.save_figure("ls_vs_logistic.pdf")

if __name__ == "__main__":
    utils.set_style()
    generate()
