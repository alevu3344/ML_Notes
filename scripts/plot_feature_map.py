"""Generate feature map plot."""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_utils as utils

def generate():
    from sklearn.datasets import make_circles
    
    X, y = make_circles(n_samples=200, noise=0.05, factor=0.4, random_state=42)
    X_class0 = X[y == 0]
    X_class1 = X[y == 1]
    Z = X[:, 0]**2 + X[:, 1]**2
    
    fig = plt.figure(figsize=(12, 6))
    
    # 2D Plot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(X_class0[:, 0], X_class0[:, 1], c=utils.COLORS['primary'], 
                edgecolor='k', label='Class 0')
    ax1.scatter(X_class1[:, 0], X_class1[:, 1], c=utils.COLORS['red'], 
                edgecolor='k', label='Class 1')
    ax1.set_title('Original 2D Data (Not Linearly Separable)')
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.5)
    
    # 3D Plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(X_class0[:, 0], X_class0[:, 1], Z[y == 0], 
                c=utils.COLORS['primary'], edgecolor='k', label='Class 0')
    ax2.scatter(X_class1[:, 0], X_class1[:, 1], Z[y == 1], 
                c=utils.COLORS['red'], edgecolor='k', label='Class 1')
    
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 10), 
                         np.linspace(-1.5, 1.5, 10))
    zz = np.full_like(xx, 0.5)
    
    ax2.plot_surface(xx, yy, zz, alpha=0.3, color='gray',
                     label='Separating Plane ($z=0.5$)')
    
    ax2.set_title('Transformed 3D Data (Linearly Separable)')
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_zlabel('$z = x_1^2 + x_2^2$')
    ax2.legend()
    
    utils.save_figure("feature_map_plot.pdf")

if __name__ == "__main__":
    utils.set_style()
    generate()
