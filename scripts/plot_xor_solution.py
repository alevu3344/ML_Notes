"""Generate XOR solution plot."""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_utils as utils

def generate():
    W = np.array([[1, 1], [1, 1]])
    c = np.array([0, -1])
    w = np.array([1, -2])
    b = 0
    
    def relu(z):
        return np.maximum(0, z)
    
    def forward(X):
        z = X @ W + c
        h = relu(z)
        y = h @ w + b
        return h, y
    
    resolution = 0.02
    x_min, x_max = -0.3, 1.3
    xx1, xx2 = np.meshgrid(np.arange(x_min, x_max, resolution),
                           np.arange(x_min, x_max, resolution))
    X_grid = np.array([xx1.ravel(), xx2.ravel()]).T
    
    H_grid, Y_grid = forward(X_grid)
    Y_grid_reshaped = Y_grid.reshape(xx1.shape)
    
    X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_xor = np.array([0, 1, 1, 0])
    colors = [utils.COLORS['primary'] if y == 0 else utils.COLORS['red'] for y in y_xor]
    markers = ['o' if y==0 else 's' for y in y_xor]
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Input Space
    axes[0].contourf(xx1, xx2, Y_grid_reshaped, alpha=0.5, levels=[-0.5, 0.5, 1.5],
                     colors=[utils.COLORS['light_bg'][0], utils.COLORS['light_bg'][2]])
    axes[0].contour(xx1, xx2, Y_grid_reshaped, levels=[0.5], colors=utils.COLORS['dark'], linewidths=2)
    
    for i in range(4):
        axes[0].scatter(X_xor[i, 0], X_xor[i, 1], c=colors[i], marker=markers[i], s=150, edgecolors='k', linewidths=1.5, zorder=10)
    
    axes[0].set_title('Input Space $(x_1, x_2)$\n(Non-linear Decision Boundary)', fontsize=11)
    axes[0].set_xlabel('$x_1$')
    axes[0].set_ylabel('$x_2$')
    axes[0].set_xlim(-0.3, 1.3)
    axes[0].set_ylim(-0.3, 1.3)
    axes[0].set_aspect('equal')
    
    # Hidden Space
    H_xor, _ = forward(X_xor)
    
    h_min, h_max = -0.3, 2.3
    hh1, hh2 = np.meshgrid(np.arange(h_min, h_max, resolution),
                           np.arange(h_min, h_max, resolution))
    H_grid_manual = np.array([hh1.ravel(), hh2.ravel()]).T
    Y_h_space = H_grid_manual @ w + b
    Y_h_space = Y_h_space.reshape(hh1.shape)
    
    axes[1].contourf(hh1, hh2, Y_h_space, alpha=0.5, levels=[-3, 0.5, 3],
                     colors=[utils.COLORS['light_bg'][0], utils.COLORS['light_bg'][2]])
    axes[1].contour(hh1, hh2, Y_h_space, levels=[0.5], colors=utils.COLORS['dark'], linewidths=2)
    
    axes[1].scatter(H_xor[0, 0], H_xor[0, 1], c=utils.COLORS['primary'], marker='o', s=150, edgecolors='k', linewidths=1.5, zorder=10)
    axes[1].scatter(1, 0, c=utils.COLORS['red'], marker='s', s=200, edgecolors='k', linewidths=2.5, zorder=10)
    axes[1].scatter(1, 0, c='none', marker='s', s=350, edgecolors=utils.COLORS['red'], linewidths=2, zorder=9)
    axes[1].scatter(H_xor[3, 0], H_xor[3, 1], c=utils.COLORS['primary'], marker='o', s=150, edgecolors='k', linewidths=1.5, zorder=10)
    
    axes[1].annotate(r'$(0,0) \to (0,0)$', xy=(0, 0), xytext=(0.3, 0.5), fontsize=8, color=utils.COLORS['dark'],
                     arrowprops=dict(arrowstyle='->', color=utils.COLORS['dark'], lw=0.8))
    axes[1].annotate(r'$(0,1), (1,0) \to (1,0)$', xy=(1, 0), xytext=(1.15, 0.5), fontsize=8, color=utils.COLORS['dark'],
                     arrowprops=dict(arrowstyle='->', color=utils.COLORS['dark'], lw=0.8))
    axes[1].annotate(r'$(1,1) \to (2,1)$', xy=(2, 1), xytext=(1.6, 0.55), fontsize=8, color=utils.COLORS['dark'],
                     arrowprops=dict(arrowstyle='->', color=utils.COLORS['dark'], lw=0.8))
    
    axes[1].set_title(r'Hidden Space $(h_1, h_2)$' + '\n(Linear Decision Boundary!)', fontsize=11)
    axes[1].set_xlabel(r'$h_1 = \max(0, x_1+x_2)$')
    axes[1].set_ylabel(r'$h_2 = \max(0, x_1+x_2-1)$')
    axes[1].set_xlim(-0.3, 2.5)
    axes[1].set_ylim(-0.3, 1.3)
    axes[1].set_aspect('equal')
    
    utils.save_figure("xor_solution_enhanced.pdf")

if __name__ == "__main__":
    utils.set_style()
    generate()
