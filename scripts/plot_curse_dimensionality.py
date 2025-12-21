"""Generate curse of dimensionality plot."""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_utils as utils

def generate():
    from scipy.special import gamma
    
    dims = np.arange(1, 21)
    
    vol_sphere = (np.pi**(dims / 2.0)) / gamma(dims / 2.0 + 1)
    vol_cube = 2.0**dims
    ratios = vol_sphere / vol_cube
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(dims, ratios, 'o-', color=utils.COLORS['primary'], 
            linewidth=2, markersize=6, label=r'Ratio: $V_{sphere} / V_{cube}$')
    
    ratio_d10 = ratios[9]
    ax.annotate(
        rf'Ratio at $d=10$ is {ratio_d10:.2e}',
        xy=(10, ratio_d10),
        xytext=(10, 0.1),
        arrowprops=dict(facecolor=utils.COLORS['dark'], shrink=0.05, width=1.5, headwidth=8),
        ha='center', fontsize=10, color=utils.COLORS['dark']
    )
    
    ax.set_yscale('log')
    ax.set_xlabel(r'Number of Dimensions ($d$)')
    ax.set_ylabel(r'Volume Ratio (Log Scale)')
    ax.set_xticks(dims[::2])
    ax.grid(True, which="both", linestyle='--', alpha=0.4)
    ax.legend()
    
    utils.save_figure("curse_dimensionality.pdf")

if __name__ == "__main__":
    utils.set_style()
    generate()
