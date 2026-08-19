Hyperspectral Image Classification using SVM (Indian Pines & Salinas)

---
Project Overview
This project applies Support Vector Machines (SVM) to pixel-wise land-cover classification of hyperspectral remote-sensing images. Two datasets are covered:
Indian Pines — evaluated in the accompanying project report, with SVM hyperparameters tuned via `GridSearchCV`.
Salinas — evaluated in the notebook `T_7_SVC_SVM_FIXED.ipynb`, with SVM hyperparameters set manually.
In both cases, the hyperspectral cube is flattened into a tabular pixel dataset, explored visually, reduced in dimensionality with PCA, and classified using SVMs with three different kernels: `rbf`, `sigmoid`, and `poly`. Results are compared by accuracy, confusion matrix, classification report, and a full-scene classification map.
---
Datasets
Dataset	Image file	Ground truth file	Shape	Classes	Pixels
Indian Pines	`Indian_pines_corrected.mat`	`Indian_pines_gt.mat`	145 × 145 × 200	17 (incl. background)	21,025
Salinas	`Salinas_corrected.mat`	`Salinas_gt.mat`	512 × 217 × 204	17 (incl. background)	111,104
Both datasets have a highly imbalanced class distribution, with the background class making up roughly half of all pixels while several land-cover classes contain only a few dozen to a few hundred samples.
---
Pipeline Structure
Data Loading — read the `.mat` files (uploaded directly for Indian Pines, loaded from Google Drive for Salinas)
Data Visualization — display a spectral band and the ground truth map
Tabular Dataset Construction — flatten the 3D cube into a DataFrame (one column per band + a `class` column), saved as `Dataset.csv`
Exploratory Data Analysis (EDA)
Spectral signature plot for a random pixel
Class distribution (counts + annotated percentage bar plot)
Single-pixel visualization as a 2D grid
Box plot and distribution plot for a user-chosen band
Dimensionality Reduction — `PCA(n_components=50, whiten=True)`, followed by an 85/15 train/test split (`random_state=11`), keeping the original pixel indices for later reconstruction of the classification map
SVM Classification with three kernels:
RBF
Sigmoid
Polynomial
Evaluation — accuracy, confusion matrix, classification report for each kernel
Classification Map — predictions mapped back onto the original image layout and visualized for each model
Final Comparison — side-by-side accuracy for all three kernels
---
Results
Indian Pines (hyperparameters tuned via `GridSearchCV` / `RandomizedSearchCV`)
SVM Kernel	Best Hyperparameters	Accuracy
RBF	`C=10`, `gamma=0.01`	0.7790
Sigmoid	`C=100`, `gamma=0.001`	0.7242
Polynomial	`degree=2`, `gamma=0.1`, `coef0=0`, `C=1`	0.7651
Salinas (fixed hyperparameters)
SVM Kernel	Hyperparameters	Accuracy
RBF	`degree=11`, `gamma='scale'` (default `C=1`)	91.47%
Sigmoid	`degree=8`, `gamma='scale'` (default `C=1`)	72.26%
Polynomial	`C=2.0`, `degree=3`, `gamma='scale'`	90.49%
Comparison
In both datasets, the RBF kernel achieves the highest accuracy, followed by the polynomial kernel, with sigmoid trailing behind in both cases.
Salinas achieves substantially higher accuracy overall (~90%+ for RBF/poly) compared to Indian Pines (~77–78%). This is consistent with Salinas being a larger, less noisy dataset with more training pixels per class after the 85/15 split (over 94,000 training pixels vs. ~17,900 for Indian Pines).
The relative gap between RBF and sigmoid is much larger for Salinas (≈19 percentage points) than for Indian Pines (≈5 points), suggesting the sigmoid kernel scales worse with the larger, more separable Salinas feature space than with Indian Pines.
Indian Pines results come from an exhaustive/randomized hyperparameter search, while Salinas results use fixed hyperparameters chosen manually — so the comparison above reflects differences in both the datasets and the tuning approach used for each.
---
Requirements
```
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
scikit-plot
keras
tqdm
```
Quick install:
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn scikit-plot keras tqdm
```
> The Salinas notebook is written for **Google Colab** with Google Drive mounted (`data_path = '/content/drive/My Drive/Colab'`). For local execution, update `data_path` to a local folder containing the `.mat` files and remove the `drive.mount(...)` call. The Indian Pines workflow instead uses direct file upload via `google.colab.files`.
---
How to Run
Make the relevant `.mat` files available (upload for Indian Pines, or place in the Google Drive path for Salinas).
Run the notebook cells in order:
Load and visualize the raw data
Build and save `Dataset.csv`
Explore spectral signatures, class balance, and per-band statistics
Apply PCA (50 components) and split into train/test sets
Train and evaluate each SVM kernel (rbf → sigmoid → poly)
Compare final accuracies
---
Output Files
```
Dataset.csv                      # Flattened pixel-level dataset
ground_truth.png                 # Saved ground truth visualization
Classification_map_rbf.png       # Classification map — RBF kernel
Classification_map_sigmoid.png   # Classification map — Sigmoid kernel
Classification_map_poly.png      # Classification map — Polynomial kernel
```
---
Notes
Both datasets show significant class imbalance; accuracy alone does not fully capture performance for minority classes, so results should be read together with the per-class precision/recall in each classification report.
The Indian Pines results rely on cross-validated hyperparameter search (`GridSearchCV`/`RandomizedSearchCV`), giving a more rigorous estimate of each kernel's best achievable accuracy. Applying the same tuning approach to Salinas would allow a fully like-for-like comparison between the two datasets.
