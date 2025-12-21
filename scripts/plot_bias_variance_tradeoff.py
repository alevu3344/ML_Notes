"""Generate bias-variance tradeoff plot."""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_utils as utils

def generate():
    complexity = np.linspace(0.05, 0.95, 100)
    bias_sq = (1 - complexity)**2
    variance = 0.05 + complexity**2 * 0.4
    irreducible_error = np.full_like(complexity, 0.1)
    total_error = bias_sq + variance + irreducible_error
    optimal_idx = np.argmin(total_error)
    optimal_complexity = complexity[optimal_idx]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(complexity, bias_sq, '--', color=utils.COLORS['primary'], linewidth=2, 
             label=r'Bias$^2$')
    ax.plot(complexity, variance, '-.', color=utils.COLORS['red'], linewidth=2, 
             label='Variance')
    ax.plot(complexity, total_error, '-', color=utils.COLORS['green'], linewidth=3, 
             label='Total Expected Error')
    ax.plot(complexity, irreducible_error, ':', color='black', linewidth=2, 
             label=r'Irreducible Error ($\sigma^2$)')
    
    ax.axvline(optimal_complexity, color='black', linestyle='--',
                label=f'Optimal Complexity')
    
    ax.text(0.2, 0.8, 'High Bias\n(Underfitting)', 
             horizontalalignment='center', fontsize=10, color=utils.COLORS['primary'])
    ax.text(0.8, 0.5, 'High Variance\n(Overfitting)', 
             horizontalalignment='center', fontsize=10, color=utils.COLORS['red'])
    
    ax.set_xlabel('Model Complexity')
    ax.set_ylabel('Expected Error')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(0, 1.5)
    ax.set_xlim(0, 1)
    
    utils.save_figure("bias_variance_tradeoff.pdf")

if __name__ == "__main__":
    utils.set_style()
    generate()
