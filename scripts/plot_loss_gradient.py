"""Generate loss gradient comparison plot."""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_utils as utils

def generate():
    z = np.linspace(-10, 10, 400)
    y_pred = 1 / (1 + np.exp(-z))
    t = 1
    
    loss_ce = -np.log(y_pred + 1e-10)
    loss_se = 0.5 * (y_pred - t)**2
    
    grad_ce = y_pred - t
    grad_se = (y_pred - t) * y_pred * (1 - y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Losses
    axes[0].plot(z, loss_ce, label='Cross-Entropy', color=utils.COLORS['primary'], linewidth=2)
    axes[0].plot(z, loss_se, label='Squared Error', color=utils.COLORS['red'], linewidth=2)
    axes[0].set_title(r'Loss Value (Target $y=1$)')
    axes[0].set_xlabel(r'$z$ (Logit)')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_ylim(0, 4)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Gradients
    axes[1].plot(z, np.abs(grad_ce), label=r'$|\nabla_z L_{CE}| = |\hat{y}-y|$', color=utils.COLORS['primary'], linewidth=2)
    axes[1].plot(z, np.abs(grad_se), label=r'$|\nabla_z L_{SE}| = |(\hat{y}-y)\hat{y}(1-\hat{y})|$', color=utils.COLORS['red'], linewidth=2)
    axes[1].set_title(r'Gradient Magnitude (Target $y=1$)')
    axes[1].set_xlabel(r'$z$ (Logit)')
    axes[1].set_ylabel('Gradient Magnitude')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    axes[1].annotate('Vanishing Gradient\n(SE)', xy=(-5, 0.02), xytext=(-5, 0.5),
                     arrowprops=dict(facecolor=utils.COLORS['red'], shrink=0.05, width=1.5),
                     ha='center', color=utils.COLORS['red'], fontsize=9)
    
    utils.save_figure("loss_gradient_comparison.pdf")

if __name__ == "__main__":
    utils.set_style()
    generate()
