# Machine and Deep Learning - Lecture Notes (DSAIT4005)

Comprehensive lecture notes for the **Machine and Deep Learning** course (DSAIT4005) at TU Delft. These notes are written in LaTeX.


# Topics Covered


- Generative vs. Discriminative Classifiers (QDA, LDA, NMC)

- Non-Parametric Methods (KDE, k-NN)

- Linear Classifiers (Perceptron, Logistic Regression)

- Bias-Variance Tradeoff

- Non-Linear Classifiers (SVMs, Decision Trees, Ensembles)

- Model Evaluation & Complexity

- Regularization

## Prerequisites

To build these notes, you need a LaTeX distribution and a Python environment.

*   **LaTeX**: TeX Live (recommended) or MiKTeX. Must include `latexmk` and `pythontex`.
*   **Python**: Python 3.10+.
*   **uv**: An extremely fast Python package installer and resolver. [Install uv](https://github.com/astral-sh/uv).

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/alevu3344/ML_Notes.git
    cd ML_Notes
    ```

2.  **Create the Python environment**:
    This project uses `uv` to manage dependencies.
    ```bash
    uv venv
    uv add numpy matplotlib scipy scikit-learn
    ```

## Building the Notes

The project includes a `.latexmkrc` file that configures the build chain (PDFLaTeX -> PythonTeX -> PDFLaTeX).

Simply run:

```bash
latexmk -pdf main.tex
```

To clean up auxiliary files:

```bash
latexmk -c
```

## Author

**Alessandro Valmori**
*   Course: DSAIT4005 - Machine and Deep Learning
*   Institution: TU Delft

---
*Note: This project uses the Roboto font family. Ensure it is installed in your TeX distribution.*
