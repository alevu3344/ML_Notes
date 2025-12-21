"""Generate sigmoid function plot."""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_utils as utils

def generate():
    z = np.linspace(-10, 10, 400)
    p = 1.0 / (1.0 + np.exp(-z))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(z, p, color=utils.COLORS['primary'], linewidth=3, label='Sigmoid Function')
    
    ax.axhline(0.5, color=utils.COLORS['dark'], linestyle='--', linewidth=1, 
               label=r'Decision Threshold ($p=0.5$)')
    ax.axvline(0, color=utils.COLORS['dark'], linestyle='--', linewidth=1,
               label=r'Decision Boundary ($z=0$)')
    
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(0.0, color='gray', linestyle=':', alpha=0.5)
    
    ax.annotate(r'Class 1 ($p > 0.5$)', xy=(5, 0.9), xytext=(7, 0.7),
                arrowprops=dict(facecolor=utils.COLORS['dark'], shrink=0.05, width=1, headwidth=8),
                fontsize=12, color=utils.COLORS['primary'], ha='center')
    
    ax.annotate(r'Class 0 ($p < 0.5$)', xy=(-5, 0.1), xytext=(-7, 0.3),
                arrowprops=dict(facecolor=utils.COLORS['dark'], shrink=0.05, width=1, headwidth=8),
                fontsize=12, color=utils.COLORS['red'], ha='center')
    
    ax.set_xlabel(r'Linear Output ($z = \mathbf{w}^T\mathbf{x} + w_0$)')
    ax.set_ylabel(r'Posterior Probability ($\hat{p}(y=1|\mathbf{x})$)')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    utils.save_figure("sigmoid_function.pdf")

if __name__ == "__main__":
    utils.set_style()
    generate()
