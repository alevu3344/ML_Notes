"""Generate activation functions plot."""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_utils as utils

def generate():
    z = np.linspace(-4, 4, 200)
    
    relu = np.maximum(0, z)
    leaky_relu = np.where(z > 0, z, 0.1 * z)
    sigmoid = 1 / (1 + np.exp(-z))
    tanh = np.tanh(z)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Activation Functions
    axes[0].plot(z, relu, label=r'ReLU: $\max(0, z)$', color=utils.COLORS['primary'], linewidth=2.5)
    axes[0].plot(z, leaky_relu, label=r'Leaky ReLU: $\max(0.1z, z)$', color=utils.COLORS['cyan'], linewidth=2, linestyle='--')
    axes[0].plot(z, sigmoid, label=r'Sigmoid: $\sigma(z)$', color=utils.COLORS['red'], linewidth=2)
    axes[0].plot(z, tanh, label='Tanh', color=utils.COLORS['green'], linewidth=2)
    
    axes[0].axhline(0, color='gray', linewidth=0.5)
    axes[0].axvline(0, color='gray', linewidth=0.5)
    axes[0].set_title('Common Activation Functions')
    axes[0].set_xlabel('$z$ (pre-activation)')
    axes[0].set_ylabel('$g(z)$')
    axes[0].legend(loc='upper left')
    axes[0].set_ylim(-1.5, 4)
    axes[0].grid(True, linestyle=':', alpha=0.5)
    
    # Derivatives
    d_relu = np.where(z > 0, 1, 0).astype(float)
    d_leaky_relu = np.where(z > 0, 1, 0.1)
    d_sigmoid = sigmoid * (1 - sigmoid)
    d_tanh = 1 - tanh**2
    
    axes[1].plot(z, d_relu, label="ReLU'", color=utils.COLORS['primary'], linewidth=2.5)
    axes[1].plot(z, d_leaky_relu, label="Leaky ReLU'", color=utils.COLORS['cyan'], linewidth=2, linestyle='--')
    axes[1].plot(z, d_sigmoid, label="Sigmoid'", color=utils.COLORS['red'], linewidth=2)
    axes[1].plot(z, d_tanh, label="Tanh'", color=utils.COLORS['green'], linewidth=2)
    
    axes[1].axhline(0, color='gray', linewidth=0.5)
    axes[1].axvline(0, color='gray', linewidth=0.5)
    axes[1].set_title('Derivatives (Crucial for Backpropagation)')
    axes[1].set_xlabel('$z$ (pre-activation)')
    axes[1].set_ylabel("$g'(z)$")
    axes[1].legend(loc='upper right')
    axes[1].set_ylim(-0.1, 1.5)
    axes[1].grid(True, linestyle=':', alpha=0.5)
    
    utils.save_figure("activation_functions.pdf")

if __name__ == "__main__":
    utils.set_style()
    generate()
