# Multi-Label Classification and K-Means Clustering on Anuran Calls Dataset  
**Vu Nguyen, USCID: 2120314402**

## Overview
This assignment performs multi-label classification and clustering on the **Anuran Calls (MFCCs) Data Set**. It utilizes **Support Vector Machines (SVM)** to classify species, genus, and family labels, and **K-means clustering** to analyze label distribution patterns. The primary focus is on evaluating the model using various multi-label metrics and optimizing model parameters for improved performance.

- **Multi-Label Classification**: Trains individual SVMs for each label using binary relevance, with Gaussian kernels and penalty weights tuned via cross-validation.
- **K-Means Clustering**: Assesses label distributions within clusters and calculates Hamming distances to evaluate clustering quality.

## Table of Contents
- Overview
- Project Structure
- Datasets
- Installation
- Methods
  - 1. Multi-Class and Multi-Label Classification Using Support Vector Machines
  - 2. K-Means Clustering on a Multi-Class and Multi-Label Data Set
- Results
- License

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
- **Anuran Calls Dataset**: Available on the UCI Machine Learning Repository

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

### 1. Multi-Class and Multi-Label Classification Using Support Vector Machines

**Goal**: To classify Family, Genus, and Species labels using a multi-class, multi-label approach with SVMs.

1. **Data Preparation**: The Anuran Calls dataset was downloaded from the UCI repository, with 70% of the data randomly allocated to the training set.  
   Each instance has three labels: Family, Genus, and Species, each with multiple classes, creating a multi-class and multi-label classification problem.

2. **Multi-Label Classification Approach**:
   - **Binary Relevance**: A separate classifier was trained for each label.
   - **Metrics**: Exact Match Score, Hamming Loss, and Hamming Score were used as evaluation metrics.

3. **Gaussian SVMs with Hyperparameter Tuning**:
   - **Configuration**: SVMs were trained with Gaussian kernels using a one-vs-all strategy.
   - **Hyperparameter Tuning**: 10-fold cross-validation was used to determine the weight of the SVM penalty (\( C \)) and kernel width (\( \gamma \)). Broad parameter ranges were initially tested, which were then narrowed based on achieving a 70% training accuracy threshold to avoid computational inefficiency.

4. **L1-Penalized SVMs**:
   - **L1 Penalty and Linear Kernel**: L1-penalized SVMs with a linear kernel were applied to encourage feature sparsity and simplify model interpretability.

5. **Handling Class Imbalance with SMOTE**:
   - **SMOTE** was used to address class imbalance, and we observed improved performance on minority classes as a result.

6. **Classifier Chain Method**:
   - The **Classifier Chain** method was applied as an alternative approach to binary relevance. It allowed us to capture inter-label dependencies, leading to comparable results while highlighting dependencies between species, genus, and family labels.

7. **Additional Multi-Label Metrics**:
   - Confusion matrices, Precision, Recall, ROC, and AUC scores were computed for each classifier to evaluate model performance across all labels in a comprehensive manner.

### 2. K-Means Clustering on a Multi-Class and Multi-Label Data Set

**Monte-Carlo Simulation**: Procedures were repeated 50 times to ensure stability, with Hamming Distances calculated.

1. **K-means Clustering**:
   - **Parameter Selection**: K values from 1 to 50 were tested, with silhouette scores guiding the selection of the optimal \( k \) for clustering.
   - **Label Distribution**: For each cluster, the majority label was determined for Family, Genus, and Species.

2. **Cluster Evaluation**:
   - Each cluster was assigned a majority label triplet, capturing dominant label distributions within clusters.
   - **Hamming Metrics**: Average Hamming Distance, Hamming Score, and Hamming Loss were calculated to evaluate the quality of clustering and label assignment accuracy.

## Results

### 1. Multi-Label Classification
- **Exact Match and Hamming Metrics**: SMOTE improved performance, particularly in imbalanced classes where minority classes gained better representation.
- **ROC AUC, Precision, and Recall**: Macro-averaged Precision, Recall, and ROC AUC scores were highest with standardized features, which consistently led to performance improvements.
- **Classifier Chains**: This approach captured inter-label dependencies more effectively, providing valuable insights beyond what binary relevance could achieve on its own.

### 2. K-Means Clustering
- **Optimal Clustering**: Silhouette scores identified the optimal \( k \) value for clustering. Clustering performance, as measured by Hamming metrics, indicated that the clusters could adequately reflect label structures.

## Conclusion
- **Classification Models**: Binary relevance SVM models with Gaussian kernels effectively handled multi-label classification, with performance gains observed through SMOTE and standardization.
- **Clustering Analysis**: K-means clustering highlighted useful distribution patterns across labels, particularly when evaluating label groupings via majority triplets and Hamming metrics.

## Additional Notes
- **Evaluation Metrics**: Added multi-label metrics, including **confusion matrices**, **precision**, **recall**, **ROC**, and **AUC** scores for a comprehensive assessment.
- **L1-Penalized SVM**: Implemented **GridSearchCV** with **10-fold cross-validation** to optimize penalty weights for L1-penalized SVM with a linear kernel.
- **Classifier Chain**: Implemented as an alternative to binary relevance to capture interdependencies across species, genus, and family labels.
- **SMOTE for Class Imbalance**: Used **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance, leading to improved classification performance.
- **Optimal Clustering (\( k \))**: Selected optimal \( k \) value using silhouette scores to evaluate clustering accuracy.
- **Monte Carlo Simulation**: Conducted **50 trials** to calculate **Hamming distances** between true labels and cluster-assigned labels, providing an average and standard deviation for robust evaluation.

## License
This project is licensed under the MIT License.
