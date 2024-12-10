# **Transfer Learning for Image Classification**

**Author:** Vu Nguyen  
**USCID:** 2120314402  

This project explores **transfer learning** for classifying images into six scene categories: `street`, `buildings`, `forest`, `glacier`, `mountain`, and `sea`. By leveraging pre-trained deep learning models (`ResNet50`, `ResNet101`, `EfficientNetB0`, `VGG16`) on the ImageNet dataset, we extract features and train custom classifiers. Data augmentation, regularization, and a systematic evaluation process are employed to identify the best-performing model.

---

## **Project Structure**
```plaintext
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

## **Getting Started**
**Google Colab Notebook**: [Link](https://colab.research.google.com/drive/1G45x1v3oIB09CpeJbvtM-oX95mUH8S3m?usp=sharing)

### **Prerequisites**
Install required libraries:
```bash
pip install -r requirements.txt
```

## **Transfer Learning Approach**

This project employs pre-trained models (`ResNet50`, `ResNet101`, `EfficientNetB0`, `VGG16`) trained on ImageNet for feature extraction:

- **Frozen Layers**: Convolutional layers are frozen to retain learned representations and reduce training time.
- **Custom Classifier**:  
  - A fully connected layer with 256 units (ReLU activation) adapts features for the specific classification task.
  - **Regularization Techniques**:
    - **L2 Regularization**: Prevents overfitting by penalizing large weights.
    - **Batch Normalization**: Stabilizes and speeds up training by normalizing activations.
    - **Dropout (20%)**: Randomly drops neurons to reduce overfitting.
  - **Output Layer**: A dense layer with softmax activation for probability distribution across the six classes.

---

## **Results and Insights**

**Performance Comparison on Test Set**:

| **Model**          | **Precision** | **Recall** | **AUC**   | **F1 Score** | **Test Accuracy** |
|---------------------|---------------|------------|-----------|--------------|-------------------|
| **ResNet50**        | 88.88%        | 88.80%     | 0.9878    | 88.74%       | 88.80%            |
| **ResNet101**       | 88.61%        | 88.80%     | 0.9291    | 88.61%       | 88.80%            |
| **EfficientNetB0**  | 82.43%        | 81.97%     | 0.8956    | 81.49%       | 81.97%            |
| **VGG16**           | 82.10%        | 80.93%     | 0.8873    | 80.70%       | 80.93%            |

**Recommendations:**
1. Deploy **ResNet50** as the primary model due to its robust performance.
2. Explore ensemble methods combining **ResNet50** and **ResNet101**.

---

## **License**
This project is licensed under the MIT License.
