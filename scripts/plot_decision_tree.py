"""Generate decision tree plot."""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_utils as utils

def generate():
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    
    iris = load_iris()
    X = iris.data[iris.target != 0, 2:]
    y = iris.target[iris.target != 0]
    
    feature_names = [iris.feature_names[2], iris.feature_names[3]]
    class_names = [iris.target_names[1], iris.target_names[2]]
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Decision Boundary
    ax = axes[0]
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    colors_region = [utils.COLORS['light_bg'][2], utils.COLORS['light_bg'][0]]
    ax.contourf(xx, yy, Z, colors=colors_region, alpha=0.8)
    
    colors_data = [utils.COLORS['green'], utils.COLORS['primary']]
    for i, color in enumerate(colors_data):
        idx = (y == i + 1)
        ax.scatter(X[idx, 0], X[idx, 1], c=color, s=40, 
                    edgecolor='k', label=class_names[i])
    
    ax.set_title('Decision Tree Boundary (max_depth=3)')
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Plot Tree Structure
    ax = axes[1]
    plot_tree(clf, 
              ax=ax,
              filled=True, 
              rounded=True,
              feature_names=feature_names,
              class_names=class_names,
              impurity=True,
              proportion=True,
              fontsize=8)
    
    ax.set_title('Learned Decision Tree Structure')
    
    utils.save_figure("decision_tree_plot.pdf")

if __name__ == "__main__":
    utils.set_style()
    generate()
