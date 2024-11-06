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

| Classifier                   | Family Metrics                                      | Genus Metrics                                   | Species Metrics                                    | Overall Multi-Label Metrics                                     |
|------------------------------|-----------------------------------------------------|-------------------------------------------------|----------------------------------------------------|-----------------------------------------------------------------|
| **1. SVM (RBF Kernel)**      | **Precision** = $0.9941$, **Recall** = $0.9694$, **ROC AUC** = $0.9970$ | **Precision** = $0.9822$, **Recall** = $0.9434$, **ROC AUC** = $0.9872$ | **Precision** = $0.9750$, **Recall** = $0.9479$, **ROC AUC** = $0.9885$ | **Exact Match Score** = $0.9838$, **Hamming Loss** = $0.0117$, **Hamming Score** = $0.9883$ |
| **2. L1-penalized SVMs**     | **Precision** = $0.7685$, **Recall** = $0.9039$, **ROC AUC** = $0.9774$ | **Precision** = $0.8365$, **Recall** = $0.8966$, **ROC AUC** = $0.9869$ | **Precision** = $0.9011$, **Recall** = $0.9105$, **ROC AUC** = $0.9898$ | **Exact Match Score** = $0.9064$, **Hamming Loss** = $0.0594$, **Hamming Score** = $0.9406$ |
| **3. SVM with SMOTE**        | **Precision** = $0.7441$, **Recall** = $0.9159$, **ROC AUC** = $0.9687$ | **Precision** = $0.7356$, **Recall** = $0.9163$, **ROC AUC** = $0.9792$ | **Precision** = $0.8742$, **Recall** = $0.9199$, **ROC AUC** = $0.9859$ | **Exact Match Score** = $0.8485$, **Hamming Loss** = $0.0790$, **Hamming Score** = $0.9210$ |
| **4. Classifier Chain**      | Best Parameters: $C=1.0$                            | **Macro Precision** = $0.83$, **Macro Recall** = $0.77$, **Macro F1-Score** = $0.80$ | **Micro Precision** = $0.95$, **Micro Recall** = $0.94$, **Micro F1-Score** = $0.95$ | **Exact Match Score** = $0.8953$, **Hamming Loss** = $0.0302$, **ROC AUC (Macro)** = $0.9588$ |

#### Key Takeaways

- **SVM (RBF Kernel)** demonstrates high consistency across all metrics, effectively handling hierarchical classification tasks with minimal errors.
- **L1-penalized SVMs** show label-specific adaptability with varying optimal $C$ values, achieving balanced performance across labels.
- **SVM with SMOTE** improves recall in minority classes, though slight precision reduction occurs, making it suitable for tasks needing balanced representation.
- **Classifier Chain** effectively captures inter-label dependencies, showing high accuracy but some difficulty in rare classes, indicating potential for further optimization.

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

#### [AI Assistance 1](https://chatgpt.com/c/67279a00-4460-8001-8944-ca86b84014cb)

##### Code Checks and Updates
- **Code Review**: Verified code against requirements for multi-label classification and clustering on the Anuran Calls dataset.
- **Evaluation Metrics**: Added metrics for multi-label classification, including **confusion matrices**, **precision**, **recall**, **ROC**, and **AUC**.
- **L1-Penalized SVM**: Implemented **GridSearchCV** with **10-fold cross-validation** to optimize the penalty weight for L1-penalized SVM with a linear kernel.

##### Classifier Chain and Class Imbalance Handling
- **Classifier Chain**: Applied the Classifier Chain method as an alternative to binary relevance for species, genus, and family labels.
- **SMOTE for Class Imbalance**: Used **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance and evaluated its effect on performance.

##### K-Means Clustering and Monte Carlo Simulation
- **Optimal Clustering (k)**: Selected optimal **k** using silhouette scores for clustering analysis on the multi-label dataset.
- **Monte Carlo Simulation**: Conducted **50 trials** to calculate **Hamming distances** between true labels and cluster-assigned labels, recording average and standard deviation.

##### README Documentation
- **Project Documentation**: Compiled a README file with:
  - **Overview**, **methods**, and **results** for both multi-label classification and K-means clustering.
  - Insights on **SMOTE** benefits for imbalanced data, feature scaling impact, and clustering quality using Hamming distance metrics.
  - **Installation** instructions and structured summary for ease of understanding.
  
#### [AI Assistance 2](https://chatgpt.com/c/6727bf0d-c394-8001-a5b9-872a3db6939c)

1. **Scaling Effects in K-means Clustering**:
   - We examined how different scaling methods impact clustering results in K-means based on socks and computer purchases:
     - **Raw Counts**: Clustering is dominated by the higher quantity item (socks).
     - **Standardized Counts**: Both products contribute equally after scaling by standard deviation, leading to balanced clusters.
     - **Dollar Values**: Clustering is dominated by the higher-value item (computers), as their high dollar values outweigh sock purchases.
   - This analysis highlighted the importance of scaling methods in similarity measures, as they impact how Euclidean distances reflect variable importance.

2. **Matrix Approximation with Principal Component Analysis (PCA)**:
   - We explored how principal component score vectors $z_{im}$ (if known) can be used to determine loading vectors $\phi_{jm}$ by performing least squares regressions with each feature in the data matrix as the response.
   - Each feature $x_j$ is regressed individually onto the principal component scores, allowing us to approximate the data matrix optimally by minimizing residuals, as specified in the optimization problem (12.6).

3. **K-means Optimization and Proof of Equation (12.18)**:
   - **Objective Function**: The K-means objective function (12.17) is:
     $$
     \sum_{k=1}^K \sum_{i \in C_k} \sum_{j=1}^p (x_{ij} - \bar{x}_{kj})^2
     $$
   - **Proof of Equation (12.18)**: We showed that the within-cluster sum of squared pairwise distances can be expressed as:
     $$
     \frac{1}{|C_k|} \sum_{i,i' \in C_k} \sum_{j=1}^p (x_{ij} - x_{i'j})^2 = 2 \sum_{i \in C_k} \sum_{j=1}^p (x_{ij} - \bar{x}_{kj})^2
     $$
     - This result was derived by expanding $(x_{ij} - x_{i'j})^2$ and using the mean $\bar{x}_{kj}$ to simplify, canceling cross terms due to the definition of the mean.
   - **Decreasing the Objective**: We explained why the K-means algorithm (Algorithm 12.2) decreases the objective function at each step:
     - **Step 2(a)**: Updating cluster centroids as the mean minimizes the within-cluster sum of squares for each cluster.
     - **Step 2(b)**: Reassigning points to the nearest centroid further reduces or maintains the objective function.
   - This iterative process ensures that the K-means objective decreases with each iteration, converging to a local minimum.

4. **Key Notation**:
   - We summarized key terms for clarity:
     - $C_k$: a cluster,
     - $x_{ij}$: the \( j \)-th feature of observation \( i \),
     - $\bar{x}_{kj}$: the mean of feature \( j \) in \( C_k \), defined as $\bar{x}_{kj} = \frac{1}{|C_k|} \sum_{i \in C_k} x_{ij}$.

This conversation covered the impact of scaling in clustering, principal component analysis for matrix approximation, the convergence of the K-means algorithm, and provided a clear summary of key notation.

