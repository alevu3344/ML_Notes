# %%
# Standard library
from mpl_toolkits.mplot3d import Axes3D

# Third-party - numerical
import numpy as np
from scipy.special import gamma

# Third-party - plotting
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse

# Third-party - scikit-learn
from sklearn.datasets import load_iris, make_circles
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KernelDensity, NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

# %%
# File: generate_qda_plot.py


def plot_gaussian_contour(ax, mean, cov, n_std=1.0, facecolor='none', **kwargs):
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    
    if np.isclose(cov[0, 0], cov[1, 1]):
        angle = 45.0
    else:
        angle = np.degrees(0.5 * np.arctan(2 * cov[0, 1] / (cov[0, 0] - cov[1, 1])))
        if cov[0, 0] < cov[1, 1]:
            angle += 90


    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)
    

    transf = transforms.Affine2D() \
        .rotate_deg(angle) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def create_qda_visualization():

    

    n_samples = 150

    mean0 = [0, 0]
    cov0 = [[2, 0.8], [0.8, 1]] 
    
    mean1 = [4, 5]
    cov1 = [[1.5, -0.6], [-0.6, 3]] 
    

    mean2 = [0, 6]
    cov2 = [[0.5, 0], [0, 0.5]] 
    

    X0 = np.random.multivariate_normal(mean0, cov0, n_samples)
    y0 = np.zeros(n_samples, dtype=int)
    
    X1 = np.random.multivariate_normal(mean1, cov1, n_samples)
    y1 = np.ones(n_samples, dtype=int)
    
    X2 = np.random.multivariate_normal(mean2, cov2, n_samples)
    y2 = 2 * np.ones(n_samples, dtype=int)
    
    X = np.vstack((X0, X1, X2))
    y = np.hstack((y0, y1, y2))
    

    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    qda.fit(X, y)
    
    x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
    y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    

    Z = qda.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    colors_region = ['#FFDDDD', '#DDFFDD', '#DDDDFF']
    plt.contourf(xx, yy, Z, colors=colors_region, alpha=0.8)
    
    plt.contour(xx, yy, Z, colors='black', linewidths=1)
    
    colors_data = ['#CC0000', '#00AA00', '#0000CC']
    for i, color in enumerate(colors_data):
        plt.scatter(X[y == i, 0], X[y == i, 1], 
                    c=color, s=20, edgecolor='k', 
                    label=f'Class {i}')

    for i, color in enumerate(colors_data):
        plot_gaussian_contour(ax, 
                              qda.means_[i], 
                              qda.covariance_[i], 
                              n_std=1.0, 
                              edgecolor=color, 
                              linestyle='--', 
                              linewidth=2)

    plt.title('Quadratic Discriminant Analysis (QDA) Decision Boundaries', fontsize=16)
    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$x_2$', fontsize=14)
    plt.legend()
    plt.grid(linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    output_filename = "qda_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")
    

if __name__ == "__main__":
    create_qda_visualization()

# %%
# File: generate_classifier_comparison.py



def create_classifier_comparison_plot():

    n_samples = 150
    np.random.seed(42) 
    
    # Class 0
    mean0 = [0, 0]
    cov0 = [[2, 0.8], [0.8, 1]]
    X0 = np.random.multivariate_normal(mean0, cov0, n_samples)
    y0 = np.zeros(n_samples, dtype=int)
    
    # Class 1
    mean1 = [4, 5]
    cov1 = [[1.5, -0.6], [-0.6, 3]]
    X1 = np.random.multivariate_normal(mean1, cov1, n_samples)
    y1 = np.ones(n_samples, dtype=int)
    
    # Class 2
    mean2 = [0, 6]
    cov2 = [[0.5, 0], [0, 0.5]]
    X2 = np.random.multivariate_normal(mean2, cov2, n_samples)
    y2 = 2 * np.ones(n_samples, dtype=int)
    
    # Combine data
    X = np.vstack((X0, X1, X2))
    y = np.hstack((y0, y1, y2))
    
    qda = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X, y)
    lda = LinearDiscriminantAnalysis(store_covariance=True).fit(X, y)
    nmc = NearestCentroid().fit(X, y) 
    
    classifiers = {
        "QDA (Unique $\Sigma_i$)": qda,
        "LDA (Shared $\Sigma$)": lda,
        "NMC ($\Sigma = \sigma^2I$)": nmc
    }
    

    x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
    y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    mesh_data = np.c_[xx.ravel(), yy.ravel()]


    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharex=True, sharey=True)
    
    colors_data = ['#CC0000', '#00AA00', '#0000CC']
    colors_region = ['#FFDDDD', '#DDFFDD', '#DDDDFF']
    
    for ax, (name, clf) in zip(axes, classifiers.items()):

        Z = clf.predict(mesh_data)
        Z = Z.reshape(xx.shape)
        

        ax.contourf(xx, yy, Z, colors=colors_region, alpha=0.8)
        

        ax.contour(xx, yy, Z, colors='black', linewidths=1)
        

        for i, color in enumerate(colors_data):
            ax.scatter(X[y == i, 0], X[y == i, 1], 
                       c=color, s=20, edgecolor='k', 
                       label=f'Class {i}')


        if name.startswith("NMC"):
            means = clf.centroids_
            ax.scatter(means[:, 0], means[:, 1], 
                       c=colors_data, s=200, marker='X', 
                       edgecolor='k', linewidth=2, label='Class Means')

        ax.set_title(name, fontsize=16)
        ax.set_xlabel('$x_1$', fontsize=14)
    
    axes[0].set_ylabel('$x_2$', fontsize=14)
    axes[0].legend()

    fig.suptitle('Comparison of Generative Gaussian Classifiers', fontsize=20, y=1.03)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    
    # Save the figure
    output_filename = "generative_classifier_comparison.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")
    
    # Optionally show the plot
    # plt.show()

if __name__ == "__main__":
    create_classifier_comparison_plot()

# %%
# File: generate_density_plots.py


def create_density_visualization():


    np.random.seed(42)
    N = 200
    X_part1 = np.random.normal(0, 0.8, int(N * 0.6))
    X_part2 = np.random.normal(4, 1.2, int(N * 0.4))
    X = np.concatenate((X_part1, X_part2))[:, np.newaxis]
    
    X_plot = np.linspace(-4, 8, 1000)[:, np.newaxis]
    

    fig, axes = plt.subplots(nrows=3, 
                             figsize=(10, 12), 
                             sharex=True)
    

    h_wide = 2.5
    bins_wide = np.arange(X.min(), X.max() + h_wide, h_wide)
    axes[0].hist(X.ravel(), bins=bins_wide, density=True, 
                 edgecolor='k', facecolor='#DDDDFF')
    axes[0].plot(X.ravel(), np.full_like(X, -0.01), '|k', markeredgewidth=1)
    axes[0].set_title(f'Histogram (h={h_wide}): Overly Smooth (High Bias)')
    axes[0].set_ylabel('Density')


    h_small = 0.15
    bins_small = np.arange(X.min(), X.max() + h_small, h_small)
    axes[1].hist(X.ravel(), bins=bins_small, density=True, 
                 edgecolor='k', facecolor='#FFFFDD')
    axes[1].plot(X.ravel(), np.full_like(X, -0.01), '|k', markeredgewidth=1)
    axes[1].set_title(f'Histogram (h={h_small}): Noisy (High Variance)')
    axes[1].set_ylabel('Density')


    h_kde = 0.5
    kde = KernelDensity(kernel='gaussian', bandwidth=h_kde).fit(X)
    log_dens = kde.score_samples(X_plot)
    
    axes[2].plot(X_plot.ravel(), np.exp(log_dens), 
                 color='red', linewidth=2,
                 label=f'KDE (h={h_kde}, Gaussian kernel)')
    axes[2].fill_between(X_plot.ravel(), np.exp(log_dens), color='#FFDDDD', alpha=0.6)
    axes[2].plot(X.ravel(), np.full_like(X, -0.01), '|k', markeredgewidth=1,
                 label='Data Points')
    axes[2].set_title(f'Kernel Density Estimation (KDE): Smooth & Flexible')
    axes[2].set_xlabel('Feature Value (x)')
    axes[2].set_ylabel('Density')
    axes[2].legend(loc='upper left')
    
    fig.suptitle('Non-Parametric Density Estimation', fontsize=16, y=1.02)
    plt.xlim(-4, 8)
    plt.tight_layout()
    
    # Save the figure
    output_filename = "density_estimation_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    create_density_visualization()

# %%
# File: generate_knn_plots.py



def create_knn_visualization():

    n_samples = 150
    np.random.seed(42)
    
    mean0 = [0, 0]
    cov0 = [[2, 0.8], [0.8, 1]]
    X0 = np.random.multivariate_normal(mean0, cov0, n_samples)
    y0 = np.zeros(n_samples, dtype=int)
    
    mean1 = [4, 5]
    cov1 = [[1.5, -0.6], [-0.6, 3]]
    X1 = np.random.multivariate_normal(mean1, cov1, n_samples)
    y1 = np.ones(n_samples, dtype=int)
    
    mean2 = [0, 6]
    cov2 = [[0.5, 0], [0, 0.5]]
    X2 = np.random.multivariate_normal(mean2, cov2, n_samples)
    y2 = 2 * np.ones(n_samples, dtype=int)
    
    X = np.vstack((X0, X1, X2))
    y = np.hstack((y0, y1, y2))
    

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    

    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    mesh_data = np.c_[xx.ravel(), yy.ravel()]
    

    classifiers = {
        "k-NN (k=1) - High Variance": KNeighborsClassifier(n_neighbors=1),
        "k-NN (k=15) - High Bias": KNeighborsClassifier(n_neighbors=15)
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    
    colors_data = ['#CC0000', '#00AA00', '#0000CC']
    colors_region = ['#FFDDDD', '#DDFFDD', '#DDDDFF']
    
    for ax, (name, clf) in zip(axes, classifiers.items()):

        clf.fit(X_scaled, y)
        

        Z = clf.predict(mesh_data)
        Z = Z.reshape(xx.shape)
        

        ax.contourf(xx, yy, Z, colors=colors_region, alpha=0.8)
        

        ax.contour(xx, yy, Z, colors='black', linewidths=1)
        

        for i, color in enumerate(colors_data):
            ax.scatter(X_scaled[y == i, 0], X_scaled[y == i, 1], 
                       c=color, s=20, edgecolor='k', 
                       label=f'Class {i}')

        ax.set_title(name, fontsize=16)
        ax.set_xlabel('Scaled $x_1$', fontsize=14)
    
    axes[0].set_ylabel('Scaled $x_2$', fontsize=14)
    axes[0].legend()

    fig.suptitle('k-NN Classifier Decision Boundaries', fontsize=20, y=1.03)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    

    output_filename = "knn_boundaries_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    create_knn_visualization()

# %%


def hypersphere_volume(d, r=1.0):

    if d < 0:
        return 0
    return (np.pi**(d / 2.0)) / gamma(d / 2.0 + 1) * (r**d)

def create_dimensionality_visualization():

    dims = np.arange(1, 21) 
    hypercube_volume = 2.0**dims
    sphere_volumes = [hypersphere_volume(d, r=1.0) for d in dims]
    volume_ratio = sphere_volumes / hypercube_volume

    plt.figure(figsize=(10, 6))
    plt.plot(dims, volume_ratio, 'bo-', 
             label='Ratio: $V_{sphere}(r=1) / V_{cube}(L=2)$')
    
    plt.title('The Curse of Dimensionality: Volume Ratio', fontsize=16)
    plt.xlabel('Number of Dimensions ($d$)', fontsize=14)
    plt.ylabel('Volume Ratio', fontsize=14)
    plt.xticks(dims[::2])
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    
    
    plt.annotate(
        f'Ratio at d=10 is {volume_ratio[9]:.2e}',
        xy=(10, volume_ratio[9]),
        xytext=(10, 0.1),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
        fontsize=12,
        horizontalalignment='center'
    )
    
    plt.yscale('log')
    plt.ylabel('Volume Ratio (Log Scale)', fontsize=14)
    plt.title('The Curse of Dimensionality: Volume Ratio (Log Scale)', fontsize=16)
    
    
    output_filename = "dimensionality_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    create_dimensionality_visualization()

# %%

w_old = np.array([2, 0.5])  
x_i = np.array([-1.5, 2.5])  
dot_product = np.dot(w_old, x_i)
print(f"Dot product w_old · x_i: {dot_product} (≤ 0 indicates misclassification for positive sample)")
w_new = w_old + x_i
fig, ax = plt.subplots(figsize=(10, 8))


ax.arrow(0, 0, w_old[0], w_old[1], head_width=0.2, head_length=0.3, fc='blue', ec='blue', label='Old weight vector (w_old)')
ax.arrow(0, 0, x_i[0], x_i[1], head_width=0.2, head_length=0.3, fc='green', ec='green', label='Misclassified positive sample (x_i)')
ax.arrow(0, 0, w_new[0], w_new[1], head_width=0.2, head_length=0.3, fc='red', ec='red', label='New weight vector (w_new)')
ax.plot(x_i[0], x_i[1], 'go', markersize=10, label='Sample point')
x_vals = np.linspace(-4, 4, 100)


if w_old[1] != 0:
    y_vals_old = - (w_old[0] / w_old[1]) * x_vals
    ax.plot(x_vals, y_vals_old, 'b--', label='Old decision boundary')


if w_new[1] != 0:
    y_vals_new = - (w_new[0] / w_new[1]) * x_vals
    ax.plot(x_vals, y_vals_new, 'r--', label='New decision boundary')


ax.annotate('Misclassified\n(should be on positive side)', xy=(x_i[0], x_i[1]), xytext=(-3, 3),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

ax.annotate('Update: w_new = w_old + x_i\nRotates boundary to classify correctly', xy=(w_new[0]/2, w_new[1]/2), xytext=(1, -2),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

pos_point = np.array([0, 1 if w_old[1] > 0 else -1])  
if np.dot(w_old, pos_point) < 0:
    pos_point = -pos_point
ax.text(2, 2, 'Positive side', color='blue', fontsize=12)
ax.text(-2, -2, 'Negative side', color='blue', fontsize=12)

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel('Feature 1 (x1)')
ax.set_ylabel('Feature 2 (x2)')
ax.set_title('Geometric Intuition of the Perceptron Update Rule')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True)

output_filename = "perceptron_update.png"
plt.savefig(output_filename, dpi=300)
print(f"Plot saved to {output_filename}")

plt.show()

# %%


def create_fld_visualization():

    np.random.seed(42)
    n_samples = 100
    
    
    mean0 = [2, 3]
    cov0 = [[1, 0.8], [0.8, 1]]
    X0 = np.random.multivariate_normal(mean0, cov0, n_samples)
    y0 = np.zeros(n_samples, dtype=int)
    
    
    mean1 = [5, 6]
    cov1 = [[1, -0.6], [-0.6, 1]]
    X1 = np.random.multivariate_normal(mean1, cov1, n_samples)
    y1 = np.ones(n_samples, dtype=int)
    
    X = np.vstack((X0, X1))
    y = np.hstack((y0, y1))
    

    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    
    w = lda.scalings_[:, 0]
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), 
                             gridspec_kw={'height_ratios': [1, 1]})
    
    colors = ['blue', 'red']
    ax = axes[0]
    proj_bad = X[:, 0] 
    
    for i, color in enumerate(colors):
        ax.scatter(X[y == i, 0], X[y == i, 1], 
                   alpha=0.4, c=color, label=f'Class {i}')
        
        ax.hist(proj_bad[y == i], bins=20, density=True, 
                alpha=0.6, color=color, 
                histtype='stepfilled', orientation='vertical', 
                edgecolor='k', range=(0, 8))
    
    ax.axhline(0, color='black', linewidth=0.5) 
    ax.set_title('Naive Projection (onto $x_1$-axis)', fontsize=16)
    ax.set_xlabel('$x_1$ (Projection Axis)')
    ax.set_ylabel('$x_2$')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    ax = axes[1]
    proj_good = X @ w 
    
    for i, color in enumerate(colors):
        ax.scatter(X[y == i, 0], X[y == i, 1], 
                   alpha=0.4, c=color, label=f'Class {i}')
    
    
    line_x = np.linspace(0, 8, 100)
    line_y = (w[1] / w[0]) * line_x 
    
    X_center = X.mean(axis=0)
    line_y = (w[1] / w[0]) * (line_x - X_center[0]) + X_center[1]
    ax.plot(line_x, line_y, color='black', linestyle='--', 
            linewidth=2, label='FLD Projection Axis ($\mathbf{w}$)')

    ax_hist = ax.inset_axes([0.05, 0.8, 0.9, 0.15])
    min_proj, max_proj = proj_good.min(), proj_good.max()
    bins = np.linspace(min_proj, max_proj, 30)
    
    for i, color in enumerate(colors):
        ax_hist.hist(proj_good[y == i], bins=bins, density=True, 
                     alpha=0.7, color=color, 
                     edgecolor='k', label=f'Class {i} Projection')
    ax_hist.set_title('Histogram of FLD Projection')
    ax_hist.set_yticks([])
    
    ax.set_title('Fisher Linear Discriminant (FLD) Projection', fontsize=16)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(axes[0].get_xlim())
    ax.set_ylim(axes[0].get_ylim())

    plt.tight_layout()
    
    
    output_filename = "fld_projection.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    create_fld_visualization()

# %%



def plot_model_boundary(ax, clf, X, label, color, style='-'):

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    
    if hasattr(clf, 'coef_') and not hasattr(clf, 'predict_proba'):
        Z = (Z > 0.5).astype(int) 
        
    Z = Z.reshape(xx.shape)
    
    ax.contour(xx, yy, Z, levels=[0.5], colors=color, 
               linestyles=style, linewidths=2)
    
    ax.plot([], [], color=color, linestyle=style, 
            linewidth=2, label=label)

def create_least_squares_outlier_visualization():
    """
    Generates a plot showing Least Squares' sensitivity to outliers.
    """
    
    
    np.random.seed(1)
    n_samples = 50
    X0 = np.random.multivariate_normal([1, 2], [[1, 0.5], [0.5, 1]], n_samples)
    y0 = np.zeros(n_samples, dtype=int) 
    
    X1 = np.random.multivariate_normal([5, 6], [[1, 0.5], [0.5, 1]], n_samples)
    y1 = np.ones(n_samples, dtype=int) 
    
    X_clean = np.vstack((X0, X1))
    y_clean = np.hstack((y0, y1))
    
    
    X_outlier = np.array([[10, 4]]) 
    y_outlier = np.array([0])
    
    X_noisy = np.vstack((X_clean, X_outlier))
    y_noisy = np.hstack((y_clean, y_outlier))
    
    
    
    
    
    ls_clean = LinearRegression().fit(X_clean, y_clean)
    log_clean = LogisticRegression().fit(X_clean, y_clean)
    
    
    ls_noisy = LinearRegression().fit(X_noisy, y_noisy)
    log_noisy = LogisticRegression().fit(X_noisy, y_noisy)
    
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    
    
    ax = axes[0]
    ax.scatter(X_clean[y_clean == 0, 0], X_clean[y_clean == 0, 1], 
               c='blue', s=30, edgecolor='k', label='Class 0')
    ax.scatter(X_clean[y_clean == 1, 0], X_clean[y_clean == 1, 1], 
               c='red', s=30, edgecolor='k', label='Class 1')
    
    plot_model_boundary(ax, ls_clean, X_clean, 
                        'Least Squares', 'blue', '--')
    plot_model_boundary(ax, log_clean, X_clean, 
                        'Logistic Regression', 'red', '-')
    
    ax.set_title('Clean Dataset', fontsize=16)
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    
    ax = axes[1]
    ax.scatter(X_clean[y_clean == 0, 0], X_clean[y_clean == 0, 1], 
               c='blue', s=30, edgecolor='k', alpha=0.7)
    ax.scatter(X_clean[y_clean == 1, 0], X_clean[y_clean == 1, 1], 
               c='red', s=30, edgecolor='k', alpha=0.7)
    
    ax.scatter(X_outlier[:, 0], X_outlier[:, 1], 
               c='blue', s=150, marker='X', 
               edgecolor='k', label='Outlier (Class 0)')
    
    plot_model_boundary(ax, ls_noisy, X_noisy, 
                        'Least Squares (skewed)', 'blue', '--')
    plot_model_boundary(ax, log_noisy, X_noisy, 
                        'Logistic Regression (robust)', 'red', '-')
    
    ax.set_title('Dataset with Outlier', fontsize=16)
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 10)
    
    
    output_filename = "least_squares_outlier.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    create_least_squares_outlier_visualization()

# %%



def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def create_sigmoid_visualization():

    z = np.linspace(-10, 10, 400)
    p = sigmoid(z)
    
    plt.figure(figsize=(10, 6))
    plt.plot(z, p, color='blue', linewidth=3)
    
    plt.axhline(0.5, color='black', linestyle='--', label='p = 0.5 (Decision Threshold)')
    plt.axvline(0, color='black', linestyle='--', label='z = 0 (Decision Boundary)')
    plt.axhline(1.0, color='gray', linestyle=':', label='p = 1.0 (Asymptote)')
    plt.axhline(0.0, color='gray', linestyle=':', label='p = 0.0 (Asymptote)')
    
    plt.annotate('Class 1', xy=(5, 0.9), xytext=(5, 0.7),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                 fontsize=12, horizontalalignment='center')
    plt.annotate('Class 0', xy=(-5, 0.1), xytext=(-5, 0.3),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                 fontsize=12, horizontalalignment='center')
    
    plt.xlabel('Linear Output ($z = \mathbf{w}^T\mathbf{x} + w_0$)', fontsize=14)
    plt.ylabel('Posterior Probability ($\hat{p}(y=1|\mathbf{x})$)', fontsize=14)
    plt.title('The Logistic (Sigmoid) Function', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_filename = "sigmoid_function.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    create_sigmoid_visualization()

# %%



def create_bias_variance_visualization():

    complexity = np.linspace(0.05, 0.95, 100)
    bias_sq = (1 - complexity)**2
    variance = 0.05 + complexity**2 * 0.4
    irreducible_error = np.full_like(complexity, 0.1)
    total_error = bias_sq + variance + irreducible_error
    optimal_idx = np.argmin(total_error)
    optimal_complexity = complexity[optimal_idx]
    
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(complexity, bias_sq, 'b--', linewidth=2, 
             label='Bias$^2$')
    plt.plot(complexity, variance, 'r-.', linewidth=2, 
             label='Variance')
    plt.plot(complexity, total_error, 'g-', linewidth=3, 
             label='Total Expected Error')
    plt.plot(complexity, irreducible_error, 'k:', linewidth=2, 
             label='Irreducible Error ($\sigma^2$)')
    
    
    plt.axvline(optimal_complexity, color='black', linestyle='--',
                label=f'Optimal Complexity')
    
    
    plt.text(0.2, 0.8, 'High Bias\n(Underfitting)', 
             horizontalalignment='center', fontsize=12, color='blue')
    plt.text(0.8, 0.5, 'High Variance\n(Overfitting)', 
             horizontalalignment='center', fontsize=12, color='red')
    
    
    plt.xlabel('Model Complexity', fontsize=14)
    plt.ylabel('Expected Error', fontsize=14)
    plt.title('The Bias-Variance Tradeoff', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1.5)
    plt.xlim(0, 1)
    
    output_filename = "bias_variance_tradeoff.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    create_bias_variance_visualization()

# %%


def create_feature_map_visualization():

    X, y = make_circles(n_samples=200, noise=0.05, factor=0.4, random_state=42)
    
    
    X_class0 = X[y == 0]
    X_class1 = X[y == 1]
    
    
    
    Z = X[:, 0]**2 + X[:, 1]**2
    
    
    fig = plt.figure(figsize=(16, 7))
    
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(X_class0[:, 0], X_class0[:, 1], c='blue', 
                edgecolor='k', label='Class 0')
    ax1.scatter(X_class1[:, 0], X_class1[:, 1], c='red', 
                edgecolor='k', label='Class 1')
    ax1.set_title('Original 2D Data (Not Linearly Separable)', fontsize=14)
    ax1.set_xlabel('$x_1$', fontsize=12)
    ax1.set_ylabel('$x_2$', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.5)

    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    
    ax2.scatter(X_class0[:, 0], X_class0[:, 1], Z[y == 0], 
                c='blue', edgecolor='k', label='Class 0')
    ax2.scatter(X_class1[:, 0], X_class1[:, 1], Z[y == 1], 
                c='red', edgecolor='k', label='Class 1')
    
    
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 10), 
                         np.linspace(-1.5, 1.5, 10))
    zz = np.full_like(xx, 0.5) 
    
    ax2.plot_surface(xx, yy, zz, alpha=0.3, color='gray',
                     label='Separating Plane ($z=0.5$)')
    
    ax2.set_title('Transformed 3D Data (Linearly Separable)', fontsize=14)
    ax2.set_xlabel('$x_1$', fontsize=12)
    ax2.set_ylabel('$x_2$', fontsize=12)
    ax2.set_zlabel('$z = x_1^2 + x_2^2$', fontsize=12)
    ax2.legend()
    
    plt.tight_layout()
    
    
    output_filename = "feature_map_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    create_feature_map_visualization()

# %%
# File: generate_tree_plot.py



def create_tree_visualization():

    iris = load_iris()
    X = iris.data[iris.target != 0, 2:] # Features 2 & 3
    y = iris.target[iris.target != 0]   # Classes 1 & 2
    
    feature_names = [iris.feature_names[2], iris.feature_names[3]]
    class_names = [iris.target_names[1], iris.target_names[2]]
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    ax = axes[0]
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    

    colors_region = ['#DDFFDD', '#DDDDFF'] 
    ax.contourf(xx, yy, Z, colors=colors_region, alpha=0.8)
    
    colors_data = ['#00AA00', '#0000CC']
    for i, color in enumerate(colors_data):
        idx = (y == i + 1)
        ax.scatter(X[idx, 0], X[idx, 1], c=color, s=40, 
                    edgecolor='k', label=class_names[i])
    
    ax.set_title('Decision Tree Boundary (max_depth=3)', fontsize=16)
    ax.set_xlabel(feature_names[0], fontsize=14)
    ax.set_ylabel(feature_names[1], fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    ax = axes[1]
    plot_tree(clf, 
              ax=ax,
              filled=True, 
              rounded=True,
              feature_names=feature_names,
              class_names=class_names,
              impurity=True,
              proportion=True,
              fontsize=10)
    
    ax.set_title('Learned Decision Tree Structure', fontsize=16)
    
    plt.tight_layout()
    
    output_filename = "decision_tree_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    create_tree_visualization()


