"""Generate gradient stability plot."""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_utils as utils

def generate():
    T = 50
    t_steps = np.arange(T)
    
    w_vanish = 0.9
    w_stable = 1.0
    w_explode = 1.1
    
    y_vanish = w_vanish ** t_steps
    y_stable = w_stable ** t_steps
    y_explode = w_explode ** t_steps
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(t_steps, y_vanish, '-', color=utils.COLORS['primary'], linewidth=2.5, label=r'Vanishing ($\lambda=0.9$)')
    ax.plot(t_steps, y_stable, '--', color='gray', linewidth=2, label=r'Stable ($\lambda=1.0$)')
    ax.plot(t_steps, y_explode, '-', color=utils.COLORS['red'], linewidth=2.5, label=r'Exploding ($\lambda=1.1$)')
    
    ax.set_title(r'Effect of Repeated Multiplication ($\lambda^t$) Over Time')
    ax.set_xlabel('Time Steps Back ($t$)')
    ax.set_ylabel(r'Gradient Magnitude ($\lambda^t$)')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_yscale('log')
    
    utils.save_figure("gradient_stability.pdf")

if __name__ == "__main__":
    utils.set_style()
    generate()
