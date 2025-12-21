"""Generate gradient descent contour plot."""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_utils as utils

def generate():
    def loss_func(w1, w2):
        return 0.5 * (w1**2 + 10 * w2**2)
    
    w1_range = np.linspace(-3, 3, 100)
    w2_range = np.linspace(-1.5, 1.5, 100)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    L = loss_func(W1, W2)
    
    def gd_update(w, lr, grad_func):
        g = grad_func(w)
        return w - lr * g
    
    def grad_loss(w):
        return np.array([w[0], 10 * w[1]])
    
    np.random.seed(42)
    lr = 0.15
    w_init = np.array([-2.5, 1.2])
    trajectory = [w_init.copy()]
    w = w_init.copy()
    for _ in range(15):
        w = gd_update(w, lr, grad_loss)
        trajectory.append(w.copy())
    trajectory = np.array(trajectory)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    contour = ax.contour(W1, W2, L, levels=15, cmap='Blues', linewidths=0.7)
    ax.contourf(W1, W2, L, levels=15, cmap='Blues', alpha=0.3)
    ax.clabel(contour, inline=True, fontsize=7, fmt='%.1f')
    
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'o-', color=utils.COLORS['red'], markersize=6, linewidth=1.5, label='GD Path')
    ax.scatter(trajectory[0, 0], trajectory[0, 1], s=100, c=utils.COLORS['red'], marker='X', zorder=10, edgecolors='k', label='Start')
    ax.scatter(0, 0, s=150, c=utils.COLORS['green'], marker='*', zorder=10, edgecolors='k', label='Optimum')
    
    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')
    ax.set_title('Gradient Descent on a Loss Surface')
    ax.legend(loc='upper right')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.5, 1.5)
    
    utils.save_figure("gradient_descent_contour.pdf")

if __name__ == "__main__":
    utils.set_style()
    generate()
