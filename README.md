
# Transfer Learning for Image Classification

## Author
Vu Nguyen  
USCID: 2120314402  

This project explores transfer learning for classifying images into six scene categories: street, buildings, forest, glacier, mountain, and sea. By leveraging pre-trained deep learning models (ResNet50, ResNet101, EfficientNetB0, VGG16) on the ImageNet dataset, we extract features and train custom classifiers. Data augmentation, regularization, and a systematic evaluation process are employed to identify the best-performing model.

## Project Structure
```
.
├── data/
│   └── seg_train/
│       ├── buildings/
│       ├── forest/
│       ├── glacier/
│       ├── mountain/
│       ├── sea/
│       └── street/
│   └── seg_test/
│       ├── buildings/
│       ├── forest/
│       ├── glacier/
│       ├── mountain/
│       ├── sea/
│       └── street/
│   └── processed_training_data/
│       ├── buildings/
│       ├── forest/
│       ├── glacier/
│       ├── mountain/
│       ├── sea/
│       └── street/
├── notebooks/
│   └── Nguyen_Vu_Final_Project.ipynb
├── requirements.txt
└── README.md
```

## Dataset
The dataset comprises six scene categories, with separate directories for training and test sets:

### Train Classes:
- **buildings**: 2191 images  
- **forest**: 2271 images  
- **glacier**: 2404 images  
- **mountain**: 2512 images  
- **sea**: 2274 images  
- **street**: 2382 images  

### Test Classes:
- **buildings**: 437 images  
- **forest**: 474 images  
- **glacier**: 553 images  
- **mountain**: 525 images  
- **sea**: 510 images  
- **street**: 501 images  

## Getting Started

### Google Colab Notebook: [Link](#)

### Prerequisites
Install required libraries:
```bash
pip install -r requirements.txt
```

### Steps to Run the Project
1. Clone the Repository:
    ```bash
    git clone https://github.com/DSCI-552/final-project-nguyenlamvu88.git
    cd final-project-nguyenlamvu88
    ```
2. Mount Google Drive (if using Colab):
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
3. Load Dataset: Update dataset paths in the notebook:
    ```python
    train_dir = "/content/drive/My Drive/Colab Notebooks/seg_train"
    test_dir = "/content/drive/My Drive/Colab Notebooks/seg_test"
    ```
4. Run the Notebook: Execute `notebooks/Nguyen_Vu_Final_Project.ipynb`.

## Transfer Learning Approach
This project employs pre-trained models (ResNet50, ResNet101, EfficientNetB0, VGG16) trained on ImageNet for feature extraction:
- **Frozen Layers**: Convolutional layers are frozen to retain learned representations and reduce training time.
- **Custom Classifier**:
  - A fully connected layer with 256 units (ReLU activation) adapts features for the specific classification task.
- **Regularization Techniques**:
  - **L2 Regularization**: Prevents overfitting by penalizing large weights.
  - **Batch Normalization**: Stabilizes and speeds up training by normalizing activations.
  - **Dropout (20%)**: Randomly drops neurons to reduce overfitting.
- **Output Layer**: A dense layer with softmax activation for probability distribution across the six classes.

### Model Training
#### Data Augmentation:
Applied transformations (RandomFlip, RandomRotation, RandomZoom, RandomContrast) to increase dataset diversity and improve model robustness.

#### Training Parameters:
- **Batch Size**: 5 (for efficient memory usage)  
- **Epochs**: Up to 70, with early stopping after 10 epochs of no improvement.  
- **Optimizer**: Adam (learning_rate=0.001)  
- **Loss Function**: Sparse Categorical Crossentropy  
- **Metrics**: Accuracy during training; Precision, Recall, F1, and AUC for evaluation.  

#### Process:
1. **Pretrained Base Model**: Load model (with include_top=False, pooling=avg).
2. **Add Classification Layers**: Flatten → Dense(256, ReLU, L2) → BatchNormalization → Dropout(0.2) → Dense(6, softmax).
3. **Compile and Train**: Use Adam optimizer and early stopping.
4. **Evaluate on Test Set**: Compute accuracy, precision, recall, F1-score, and AUC.

### Model Evaluation
#### Evaluation Metrics:
- **Precision**: Accuracy of positive predictions.  
- **Recall**: Ability to capture all true positives.  
- **F1 Score**: Harmonic mean of precision and recall.  
- **AUC (One-vs-Rest)**: Measures class separability.  
- **Test Accuracy**: Model’s overall performance on unseen data.

#### Visualization:
Training and validation loss curves are plotted over epochs to analyze convergence and generalization.

### Results and Insights
#### Performance Comparison on Test Set:

| Model          | Precision | Recall | AUC   | F1 Score | Test Accuracy |
|----------------|-----------|--------|-------|----------|---------------|
| ResNet50       | 88.88%    | 88.80% | 0.9878| 88.74%   | 88.80%        |
| ResNet101      | 88.61%    | 88.80% | 0.9291| 88.61%   | 88.80%        |
| EfficientNetB0 | 82.43%    | 81.97% | 0.8956| 81.49%   | 81.97%        |
| VGG16          | 82.10%    | 80.93% | 0.8873| 80.70%   | 80.93%        |

### Summary
All four models share a common training pipeline with transfer learning, data augmentation, and careful regularization. The results highlight that ResNet50 stands out as the top performer, with potential future improvements through ensembling or exploring more advanced models.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Keras Documentation
- ImageNet Pre-trained Models
- TensorFlow

