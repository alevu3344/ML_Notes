"""Generate perceptron update plot."""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_utils as utils

def generate():
    w_old = np.array([2, 0.5])
    x_i = np.array([-1.5, 2.5])
    dot_product = np.dot(w_old, x_i)
    w_new = w_old + x_i
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Old weight vector
    ax.arrow(0, 0, w_old[0], w_old[1], 
             head_width=0.2, head_length=0.3, fc=utils.COLORS['primary'], ec=utils.COLORS['primary'], 
             label=r'Old weight $\mathbf{w}_{old}$', linewidth=2)
    
    # Misclassified sample vector
    ax.arrow(0, 0, x_i[0], x_i[1], 
             head_width=0.2, head_length=0.3, fc=utils.COLORS['green'], ec=utils.COLORS['green'], 
             label=r'Misclassified sample $\mathbf{x}_i$', linewidth=2)
    
    # New weight vector
    ax.arrow(0, 0, w_new[0], w_new[1], 
             head_width=0.2, head_length=0.3, fc=utils.COLORS['red'], ec=utils.COLORS['red'], 
             label=r'New weight $\mathbf{w}_{new} = \mathbf{w}_{old} + \mathbf{x}_i$', linewidth=2)
    
    ax.plot(x_i[0], x_i[1], 'o', markersize=8, color=utils.COLORS['green'], 
            markeredgecolor=utils.COLORS['dark'], zorder=5)
    
    x_vals = np.linspace(-4, 4, 100)
    
    if w_old[1] != 0:
        y_vals_old = - (w_old[0] / w_old[1]) * x_vals
        ax.plot(x_vals, y_vals_old, '--', color=utils.COLORS['primary'], 
                label=r'Old boundary $(\mathbf{w}_{old} \cdot \mathbf{x} = 0)$')
    
    if w_new[1] != 0:
        y_vals_new = - (w_new[0] / w_new[1]) * x_vals
        ax.plot(x_vals, y_vals_new, '--', color=utils.COLORS['red'], 
                label=r'New boundary $(\mathbf{w}_{new} \cdot \mathbf{x} = 0)$')
    
    ax.text(x_i[0], x_i[1] + 0.5, r'$\mathbf{x}_i$', color=utils.COLORS['green'], ha='center', fontsize=12)
    ax.text(w_old[0] + 0.2, w_old[1], r'$\mathbf{w}_{old}$', color=utils.COLORS['primary'], ha='left', fontsize=12)
    ax.text(w_new[0] + 0.2, w_new[1], r'$\mathbf{w}_{new}$', color=utils.COLORS['red'], ha='left', fontsize=12)
    
    ax.annotate(r'Misclassified for $\mathbf{w}_{old}$' + '\n' + r'($\mathbf{w}_{old} \cdot \mathbf{x}_i = ' + f'{dot_product:.2f} \\leq 0$)', 
                xy=(x_i[0], x_i[1]), xytext=(-3.5, 2.5),
                arrowprops=dict(facecolor=utils.COLORS['dark'], shrink=0.05, width=1, headwidth=6), 
                fontsize=9, color=utils.COLORS['dark'], ha='left')
    
    ax.text(3.5, 3.5, 'Positive Side', color=utils.COLORS['primary'], ha='right', va='top', fontsize=10)
    ax.text(-3.5, -3.5, 'Negative Side', color=utils.COLORS['primary'], ha='left', va='bottom', fontsize=10)
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.axhline(0, color='gray', lw=0.5, linestyle=':')
    ax.axvline(0, color='gray', lw=0.5, linestyle=':')
    ax.set_xlabel(r'Feature $x_1$')
    ax.set_ylabel(r'Feature $x_2$')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    utils.save_figure("perceptron_update.pdf")

if __name__ == "__main__":
    utils.set_style()
    generate()
