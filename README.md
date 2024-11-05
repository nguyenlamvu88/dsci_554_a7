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
- **Exact Match and Hamming Metrics**: SMOTE enhanced prediction accuracy, particularly for minority classes that were better represented.
- **ROC AUC, Precision, and Recall**: Standardizing features led to the highest macro-averaged scores, consistently improving model performance.
- **Classifier Chains**: This method effectively captured inter-label dependencies, providing insights beyond traditional binary relevance.

### 2. K-Means Clustering
- **Optimal Clustering**: The Silhouette Score determined the optimal number of clusters as \( k = 5 \).

### 3. Hamming Metrics for Clustering Performance
   - **Average Hamming Distance**: $0.1968$ suggests that about $19.68\%$ of the predicted labels do not match the true labels.
   - **Hamming Score**: $0.8032$, indicating $80.32\%$ of label predictions are correct.
   - **Hamming Loss**: $0.1968$, confirming that $19.68\%$ of label predictions were incorrect.

   These metrics indicate that while the clustering model reflects some structure in the data, there is moderate room for improvement in label accuracy.

### 4. Multi-Label Model Comparison Summary

| Classifier                   | Family Metrics                                    | Genus Metrics                                   | Species Metrics                                  | Overall Multi-Label Metrics                              |
|------------------------------|---------------------------------------------------|-------------------------------------------------|--------------------------------------------------|----------------------------------------------------------|
| **1. SVM (RBF Kernel)**      | **Precision** = $0.9941$, **Recall** = $0.9694$, **ROC AUC** = $0.9970$ | **Precision** = $0.9822$, **Recall** = $0.9434$, **ROC AUC** = $0.9872$ | **Precision** = $0.9750$, **Recall** = $0.9479$, **ROC AUC** = $0.9885$ | **Exact Match Score** = $0.9838$, **Hamming Loss** = $0.0117$ (Hamming Score = $0.9883$) |
| **2. L1-penalized SVMs**     | **Precision** = $0.7685$, **Recall** = $0.9039$, **ROC AUC** = $0.9774$ | **Precision** = $0.8365$, **Recall** = $0.8966$, **ROC AUC** = $0.9869$ | **Precision** = $0.9011$, **Recall** = $0.9105$, **ROC AUC** = $0.9898$ | **Exact Match Score** = $0.9064$, **Hamming Loss** = $0.0594$ (Hamming Score = $0.9406$) |
| **3. SVM with SMOTE**        | **Precision** = $0.7441$, **Recall** = $0.9159$, **ROC AUC** = $0.9687$ | **Precision** = $0.7356$, **Recall** = $0.9163$, **ROC AUC** = $0.9792$ | **Precision** = $0.8742$, **Recall** = $0.9199$, **ROC AUC** = $0.9859$ | **Exact Match Score** = $0.8485$, **Hamming Loss** = $0.0790$ (Hamming Score = $0.9210$) |
| **4. Classifier Chain**      | Best Parameters: $C=1.0$                          | **Macro Precision** = $0.83$, **Macro Recall** = $0.77$, **Macro F1-Score** = $0.80$ | **Micro Precision** = $0.95$, **Micro Recall** = $0.94$, **Micro F1-Score** = $0.95$ | **Exact Match Score** = $0.8953$, **Hamming Loss** = $0.0302$, **ROC AUC (Macro)** = $0.9588$ |

### Summary of Findings

- **SVM (RBF Kernel)** exhibits strong performance across all labels, with optimal parameters of ($C=35.938$, $\gamma=0.1$) and minimal errors, particularly effective for hierarchical and taxonomic tasks.

- **L1-penalized SVMs** show robust label-specific performance with varying optimal $C$ values ($C=215.44$ for Family, $C=46.42$ for Genus, $C=10.0$ for Species), achieving high accuracy but minor misclassifications, particularly in the "Family" label.

- **SVM with SMOTE** enhances recall by improving label balance in training data, though precision slightly declines due to class complexity. It is suitable for applications focused on balanced representation.

- **Classifier Chain** achieves high accuracy with strong micro-averaged metrics but struggles with lower performance in less frequent classes, indicating potential for further optimization.

--- Majority Labels for Each Cluster ---
- **Cluster 0**: Majority Family: Leptodactylidae, Majority Genus: Adenomera, Majority Species: AdenomeraHylaedactylus
- **Cluster 1**: Majority Family: Hylidae, Majority Genus: Hypsiboas, Majority Species: HypsiboasCinerascens
- **Cluster 2**: Majority Family: Leptodactylidae, Majority Genus: Adenomera, Majority Species: AdenomeraAndre
- **Cluster 3**: Majority Family: Leptodactylidae, Majority Genus: Adenomera, Majority Species: AdenomeraAndre
- **Cluster 4**: Majority Family: Hylidae, Majority Genus: Hypsiboas, Majority Species: HypsiboasCordobae

--- Hamming Metrics ---
- **Average Hamming Distance**: $0.1968$
- **Hamming Score**: $0.8032$
- **Hamming Loss**: $0.1968$

These metrics indicate that, on average, about $19.68\%$ of the label predictions are incorrect, with a score of $80.32\%$ correct predictions. This suggests that the clustering model adequately reflects some of the data's structure but has room for improvement in label accuracy.

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
