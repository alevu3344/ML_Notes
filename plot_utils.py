# plot_utils.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import os

# --- 1. TU Delft Color Palette ---
COLORS = {
    'primary': '#0076C2',       # TU Delft Blue
    'dark': '#0C2340',          # Dark Blue
    'cyan': '#00A6D6',          # Cyan
    'red': '#E03C31',           # Alert Red
    'green': '#009B77',         # Success Green
    'gray': '#A2A2A2',          # Neutral
    'light_bg': ['#E6F2FA', '#E6F9FD', '#FAEBEB'] 
}

# --- 2. Style Initialization ---
def set_style():
    """Applies the custom matplotlib style settings."""
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Roboto', 'Helvetica', 'Arial'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 9,
        'legend.frameon': False,
        'figure.figsize': (6, 4),
        'savefig.format': 'pdf',
        'lines.linewidth': 2,
        'lines.markersize': 6,
    })

# --- 3. Helper Functions ---

def save_and_include(filename, width=r"0.8\textwidth", tight=True):
    """Saves figure and prints the LaTeX includegraphics command."""
    output_dir = "generated_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filepath = os.path.join(output_dir, filename)

    if tight:
        plt.tight_layout()
    
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # This print statement is captured by PythonTeX to inject LaTeX
    print(r'\begin{center}')
    print(r'\includegraphics[width=' + width + ']{' + output_dir + '/' + filename + '}')
    print(r'\end{center}')

def plot_decision_regions(ax, clf, X, y, resolution=0.02):
    """Generic helper to plot decision regions."""
    markers = ('o', 's', '^', 'v', '<')
    colors = [COLORS['primary'], COLORS['red'], COLORS['green']]
    cmap_light = mpl.colors.ListedColormap([COLORS['light_bg'][0], COLORS['light_bg'][2], COLORS['light_bg'][1]])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    ax.contourf(xx1, xx2, Z, alpha=0.6, cmap=cmap_light)
    ax.contour(xx1, xx2, Z, colors=COLORS['dark'], linewidths=0.5, alpha=0.5)

    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y == cl, 0], 
                   y=X[y == cl, 1],
                   alpha=0.9, 
                   c=colors[idx],
                   edgecolor=COLORS['dark'],
                   marker=markers[idx], 
                   label=f'Class {cl}',
                   s=30)

def plot_gaussian_ellipse(ax, mean, cov, color, n_std=1.0):
    """Draws a Gaussian confidence ellipse."""
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor='none', edgecolor=color, linestyle='--', linewidth=1.5)
    
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    
    if np.isclose(cov[0, 0], cov[1, 1]):
        # When variances are equal, ellipse tilts at ±45° depending on covariance sign
        angle = 45.0 if cov[0, 1] >= 0 else -45.0
    else:
        angle = np.degrees(0.5 * np.arctan(2 * cov[0, 1] / (cov[0, 0] - cov[1, 1])))
        if cov[0, 0] < cov[1, 1]: angle += 90

    transf = transforms.Affine2D().rotate_deg(angle).scale(scale_x, scale_y).translate(mean[0], mean[1])
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)