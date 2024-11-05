
# Multi-Label Classification and K-Means Clustering on Anuran Calls Dataset  
**Vu Nguyen, USCID: 2120314402**

## Overview
This assignment performs multi-label classification and clustering on the **Anuran Calls (MFCCs) Data Set**. It utilizes **Support Vector Machines (SVM)** to classify species, genus, and family labels, and **K-means clustering** to analyze label distribution patterns. The primary focus is on evaluating the model using various multi-label metrics and optimizing model parameters for improved performance.

- **Multi-Label Classification**: Trains individual SVMs for each label using binary relevance, with Gaussian kernels and penalty weights tuned via cross-validation.
- **K-Means Clustering**: Assesses label distributions within clusters and calculates Hamming distances to evaluate clustering quality.

## Project Structure
```
. 
├── data/
│   └── Frogs_MFCCs.csv
├── notebook/ 
│   └── Nguyen_Vu_HW7.ipynb
├── requirements.txt
└── README.md
```

## Datasets
- **Anuran Calls Dataset**: Available on the UCI Machine Learning Repository.

## Installation
Clone the repository:
```bash
git clone https://github.com/nguyenlamvu88/homework-7-nguyenlamvu88
```

Install the required packages:
```bash
pip install -r requirements.txt
```
Alternatively, install packages directly:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost tqdm
```

## Methods

### 1. Multi-Label Classification Using Support Vector Machines
**Goal**: To classify Family, Genus, and Species labels using a multi-label approach with SVMs.

**Data Preparation**: The Anuran Calls dataset was downloaded from the UCI repository, with 70% of the data allocated to the training set. Each instance has three labels: Family, Genus, and Species, creating a multi-label classification problem.

**Multi-Label Classification Approach**:
- **Binary Relevance**: A separate classifier was trained for each label.
- **Metrics**: Exact Match Score, Hamming Loss, and Hamming Score were used for evaluation.
- **Gaussian SVMs with Hyperparameter Tuning**:
  - **Configuration**: SVMs were trained with Gaussian kernels using a one-vs-all strategy.
  - **Hyperparameter Tuning**: 10-fold cross-validation determined the SVM penalty (C) and kernel width (γ). Parameter ranges were initially broad, then narrowed based on achieving a 70% training accuracy threshold.
- **L1-Penalized SVMs**: L1-penalized SVMs with a linear kernel were applied to encourage feature sparsity and simplify model interpretability.
- **Handling Class Imbalance with SMOTE**: SMOTE was implemented to address class imbalance, enhancing performance on minority classes.
- **Classifier Chain Method**: The Classifier Chain method was applied to capture inter-label dependencies, providing insights beyond binary relevance.
- **Additional Multi-Label Metrics**: Confusion matrices, Precision, Recall, ROC, and AUC scores were computed to evaluate model performance comprehensively.

### 2. K-Means Clustering
**Monte-Carlo Simulation**: Procedures were repeated 50 times to ensure stability, with Hamming Distances calculated.

**K-means Clustering**:
- **Parameter Selection**: K values from 1 to 50 were tested, with silhouette scores guiding the selection of the optimal k.
- **Label Distribution**: The majority label was determined for Family, Genus, and Species for each cluster.

**Cluster Evaluation**:
- Each cluster was assigned a majority label triplet, capturing dominant label distributions within clusters.
- **Hamming Metrics**: Average Hamming Distance, Hamming Score, and Hamming Loss were calculated to evaluate clustering quality and label assignment accuracy.

## Results

### 1. Multi-Label Classification Metrics
| Classifier                     | Family Precision | Family Recall | Genus Precision | Genus Recall | Species Precision | Species Recall | Overall Metrics                  |
|--------------------------------|------------------|---------------|------------------|--------------|-------------------|----------------|----------------------------------|
| SVM (RBF Kernel)              | 0.9941           | 0.9694       | 0.9822           | 0.9434       | 0.9750            | 0.9479         | Exact Match Score = 0.9838<br>Hamming Loss = 0.0117<br>Hamming Score = 0.9883 |
| L1-Penalized SVMs             | 0.7685           | 0.9039       | 0.8365           | 0.8966       | 0.9011            | 0.9105         | Exact Match Score = 0.9064<br>Hamming Loss = 0.0594<br>Hamming Score = 0.9406 |
| SVM with SMOTE                | 0.7441           | 0.9159       | 0.7356           | 0.9163       | 0.8742            | 0.9199         | Exact Match Score = 0.8485<br>Hamming Loss = 0.0790<br>Hamming Score = 0.9210 |
| Classifier Chain               | -                | -             | -                | -            | -                 | -              | Exact Match Score = 0.8953<br>Hamming Loss = 0.0302<br>ROC AUC (Macro) = 0.9588 |

### 2. K-Means Clustering
- **Optimal Clustering**: The Silhouette Score determined the optimal number of clusters as k=5.

### 3. Hamming Metrics for Clustering Performance
- Average Hamming Distance: 0.1968, indicating about 19.68% of predicted labels do not match true labels.
- Hamming Score: 0.8032, indicating 80.32% of predictions are correct.
- Hamming Loss: 0.1968, confirming 19.68% of predictions were incorrect.

These metrics show that while the clustering model reflects some data structure, there is room for improvement in accuracy.

### 4. Multi-Label Model Comparison Summary
- **Summary of Findings**: SVM (RBF Kernel) performs consistently across all labels with optimal parameters (C=35.938, γ=0.1) and minimal errors, making it effective for hierarchical and taxonomic tasks.

--- Majority Labels for Each Cluster ---

- Cluster 0: Majority Family: Leptodactylidae, Majority Genus: Adenomera, Majority Species: AdenomeraHylaedactylus
- Cluster 1: Majority Family: Hylidae, Majority Genus: Hypsiboas, Majority Species: HypsiboasCinerascens
- Cluster 2: Majority Family: Leptodactylidae, Majority Genus: Adenomera, Majority Species: AdenomeraAndre
- Cluster 3: Majority Family: Leptodactylidae, Majority Genus: Adenomera, Majority Species: AdenomeraAndre
- Cluster 4: Majority Family: Hylidae, Majority Genus: Hypsiboas, Majority Species: HypsiboasCordobae

--- Hamming Metrics ---

- Average Hamming Distance: 0.1968
- Hamming Score: 0.8032
- Hamming Loss: 0.1968

These metrics show that, on average, about 19.68% of the label predictions are incorrect, reflecting moderate room for improvement in label accuracy.

## Conclusion
- **Classification Models**: Binary relevance SVM models with Gaussian kernels effectively handled multi-label classification, with performance gains observed through SMOTE and standardization.
- **Clustering Analysis**: K-means clustering highlighted useful distribution patterns across labels, particularly when evaluating label groupings via majority triplets and Hamming metrics.

## Additional Notes
- **Evaluation Metrics**: Added multi-label metrics, including confusion matrices, precision, recall, ROC, and AUC scores for comprehensive assessment.
- **L1-Penalized SVM**: Implemented GridSearchCV with 10-fold cross-validation to optimize penalty weights for L1-penalized SVM with a linear kernel.
- **Classifier Chain**: Implemented as an alternative to binary relevance to capture interdependencies across species, genus, and family labels.
- **SMOTE for Class Imbalance**: Used SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance, leading to improved classification performance.
- **Optimal Clustering (k)**: Selected optimal k value using silhouette scores to evaluate clustering accuracy.
- **Monte Carlo Simulation**: Conducted 50 trials to calculate Hamming distances between true labels and cluster-assigned labels, providing an average and standard deviation for robust evaluation.

## License
This project is licensed under the MIT License.
