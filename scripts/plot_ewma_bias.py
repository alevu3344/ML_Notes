"""Generate EWMA bias correction plot."""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_utils as utils

def generate():
    np.random.seed(42)
    T = 100
    t_arr = np.arange(1, T + 1)
    true_signal = np.ones(T)
    noise = np.random.normal(0, 0.5, T)
    y = true_signal + noise
    
    rho = 0.9
    v = np.zeros(T)
    v_corrected = np.zeros(T)
    current_v = 0
    
    for i in range(T):
        current_v = rho * current_v + (1 - rho) * y[i]
        v[i] = current_v
        v_corrected[i] = current_v / (1 - rho**(i + 1))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(t_arr, y, 'o', color=utils.COLORS['gray'], alpha=0.4, label=r'Noisy Gradients ($g_t$)', markersize=4)
    ax.plot(t_arr, v, '--', color=utils.COLORS['red'], linewidth=2, label=rf'EWMA ($\rho={rho}$, No Bias Corr.)')
    ax.plot(t_arr, v_corrected, '-', color=utils.COLORS['primary'], linewidth=2.5, label=rf'EWMA ($\rho={rho}$, Bias Corrected)')
    
    ax.axhline(1.0, color='black', linestyle=':', label='True Mean')
    
    ax.set_title('Smoothing Noisy Signals with EWMA')
    ax.set_xlabel('Iteration ($t$)')
    ax.set_ylabel('Value')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    utils.save_figure("ewma_bias_correction.pdf")

if __name__ == "__main__":
    utils.set_style()
    generate()
