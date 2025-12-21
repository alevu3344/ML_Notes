#!/usr/bin/env python3
"""
Master script to generate all plots for the LaTeX document.
Run this before compiling the LaTeX document.

Usage:
    python scripts/generate_all_plots.py
"""

import sys
import os

# Add parent directory to path so we can import plot_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import plot_utils as utils

# Apply global style
utils.set_style()

# Ensure we're in the right directory for saving plots
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all plot generation functions
from scripts.plot_qda_example import generate as gen_qda_example
from scripts.plot_generative_comparison import generate as gen_generative_comparison
from scripts.plot_density_estimation import generate as gen_density_estimation
from scripts.plot_knn_boundaries import generate as gen_knn_boundaries
from scripts.plot_curse_dimensionality import generate as gen_curse_dimensionality
from scripts.plot_perceptron_update import generate as gen_perceptron_update
from scripts.plot_fld_projection import generate as gen_fld_projection
from scripts.plot_ls_vs_logistic import generate as gen_ls_vs_logistic
from scripts.plot_sigmoid_function import generate as gen_sigmoid_function
from scripts.plot_bias_variance_tradeoff import generate as gen_bias_variance_tradeoff
from scripts.plot_feature_map import generate as gen_feature_map
from scripts.plot_decision_tree import generate as gen_decision_tree
from scripts.plot_activation_functions import generate as gen_activation_functions
from scripts.plot_xor_solution import generate as gen_xor_solution
from scripts.plot_gradient_descent import generate as gen_gradient_descent
from scripts.plot_loss_gradient import generate as gen_loss_gradient
from scripts.plot_ewma_bias import generate as gen_ewma_bias
from scripts.plot_gradient_stability import generate as gen_gradient_stability

def main():
    """Generate all plots."""
    plots = [
        ("qda_example.pdf", gen_qda_example),
        ("generative_comparison.pdf", gen_generative_comparison),
        ("density_estimation.pdf", gen_density_estimation),
        ("knn_boundaries.pdf", gen_knn_boundaries),
        ("curse_dimensionality.pdf", gen_curse_dimensionality),
        ("perceptron_update.pdf", gen_perceptron_update),
        ("fld_projection.pdf", gen_fld_projection),
        ("ls_vs_logistic.pdf", gen_ls_vs_logistic),
        ("sigmoid_function.pdf", gen_sigmoid_function),
        ("bias_variance_tradeoff.pdf", gen_bias_variance_tradeoff),
        ("feature_map_plot.pdf", gen_feature_map),
        ("decision_tree_plot.pdf", gen_decision_tree),
        ("activation_functions.pdf", gen_activation_functions),
        ("xor_solution_enhanced.pdf", gen_xor_solution),
        ("gradient_descent_contour.pdf", gen_gradient_descent),
        ("loss_gradient_comparison.pdf", gen_loss_gradient),
        ("ewma_bias_correction.pdf", gen_ewma_bias),
        ("gradient_stability.pdf", gen_gradient_stability),
    ]
    
    print(f"Generating {len(plots)} plots...")
    
    for name, generate_func in plots:
        try:
            generate_func()
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            raise
    
    print("Done!")

if __name__ == "__main__":
    main()
