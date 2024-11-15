
# Supervised, Semi-Supervised, Unsupervised, and Active Learning on Breast Cancer and Banknote Authentication Datasets  
**Vu Nguyen, USCID: 2120314402**

## Overview
This assignment applies supervised, semi-supervised, unsupervised, and active learning methods on two datasets: **Breast Cancer Wisconsin (Diagnostic)** and **Banknote Authentication**. The project explores the effectiveness of each learning approach through Monte Carlo simulations, evaluating metrics such as accuracy, precision, recall, F1-score, and AUC.

- **Supervised Learning**: L1-penalized Support Vector Machines (SVMs) with cross-validation.
- **Semi-Supervised Learning**: Self-training SVMs using an iterative labeling approach.
- **Unsupervised Learning**: K-means and Spectral Clustering.
- **Active Learning**: Incremental SVM training with passive and active data selection strategies.

## Project Structure
```plaintext
.
├── data/                          
│   └── Breast_Cancer_Dataset.csv  
│   └── Banknote_Authentication_Dataset.csv  
├── notebook/
│   └── Nguyen_Vu_HW8.ipynb  
├── requirements.txt               
└── README.md
```

## Datasets
- **Breast Cancer Wisconsin (Diagnostic) Dataset**: Available on the UCI Machine Learning Repository.
- **Banknote Authentication Dataset**: Available on the UCI Machine Learning Repository.

## Installation
Clone the repository:
```bash
git clone https://github.com/nguyenlamvu88/homework-8-nguyenlamvu88
```

Install the required packages:
```bash
pip install -r requirements.txt
```

Alternatively, install packages directly:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tqdm
```

## Methods

### 1. Supervised, Semi-Supervised, and Unsupervised Learning (Breast Cancer Dataset)

**Objective**: To analyze the effectiveness of various learning methods on a binary classification problem using Monte Carlo simulation.

1. **Supervised Learning with SVM**:
   - An L1-penalized SVM model was trained on the full labeled dataset.
   - Performance was evaluated using cross-validation and metrics including accuracy, precision, recall, F1-score, and AUC.

2. **Semi-Supervised Learning (Self-Training SVM)**:
   - 50% of each class was labeled, with the remaining data iteratively labeled by the SVM model.
   - Each iteration labeled the data point farthest from the decision boundary, improving model performance progressively.

3. **Unsupervised Learning**:
   - **K-means Clustering**: K-means clustering with \( k = 2 \) was performed multiple times, with majority polling used for label assignment.
   - **Spectral Clustering**: Spectral clustering with an RBF kernel balanced clusters based on class distribution.

### 2. Active Learning Using SVM (Banknote Authentication Dataset)

**Objective**: To examine the efficiency of passive and active learning strategies on a binary classification problem.

1. **Passive Learning**:
   - SVMs were trained with random data point additions (in steps of 10), and test errors were tracked across 90 SVMs.

2. **Active Learning**:
   - SVMs selected data points closest to the decision boundary in each iteration, adding these points for improved performance.

3. **Learning Curve Comparison**:
   - A plot of test error versus training instances compares active and passive learning strategies.

## Results

### 1. Supervised, Semi-Supervised, and Unsupervised Learning (Breast Cancer Dataset)

| **Method**                                       | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **AUC**    |
|--------------------------------------------------|--------------|---------------|------------|--------------|------------|
| **Supervised Learning**                          | 0.973        | 0.981        | 0.945      | 0.962       | 0.967      |
| **Semi-Supervised Learning**                     | 0.961        | 0.985        | 0.910      | 0.945       | 0.951      |
| **Unsupervised Learning - K-Means (Closest 30)** | 0.928        | 0.952        | 0.849      | 0.898       | 0.912      |
| **Unsupervised Learning - Spectral Clustering**  | 0.856        | 0.978        | 0.627      | 0.764       | 0.809      |

### 2. Active vs. Passive Learning (Banknote Dataset)

- **Learning Curves**: The average test error for active learning was consistently lower than for passive learning, demonstrating the efficiency of active selection in enhancing model performance.

### Key Findings

- **Supervised Learning** achieved the highest performance with access to full labeled data.
- **Semi-Supervised Learning** effectively approximated supervised performance with fewer initial labels.
- **Unsupervised Learning** (especially spectral clustering) had the lowest accuracy, emphasizing the need for labeled data in complex classification tasks.
- **Active Learning** outperformed passive learning, indicating its usefulness in training efficiency.

## Conclusion

This project demonstrates the comparative effectiveness of supervised, semi-supervised, unsupervised, and active learning on real-world datasets. Supervised learning performed best with complete data, while semi-supervised learning showed potential for label-scarce scenarios. Active learning proved more efficient than passive learning, underscoring its value in reducing model error with fewer labeled examples.

## License
This project is licensed under the MIT License.

--- 

This README provides a structured overview of the notebook, highlighting key learning approaches, dataset details, methodology, and results for effective understanding and reproducibility.
