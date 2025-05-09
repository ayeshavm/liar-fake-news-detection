# 🧠 A Hybrid Approach to Fake News Detection

**Author**: Ayesha Mendoza  
**Institution**: University of Colorado Boulder  
**Dataset**: [LIAR Dataset](https://aclanthology.org/P17-2067/)  
**Tags**: NLP · BERT · Fake News · Monte Carlo Dropout · Transformers · PyTorch

---

## 📌 Project Overview

Misinformation is one of the biggest threats in our digital age, especially in political discourse. This project explores a hybrid NLP and machine learning pipeline for classifying the truthfulness of political statements using the LIAR dataset. It compares classic and deep learning approaches, introducing a hybrid architecture that combines BERT embeddings with structured metadata and uncertainty estimation via Monte Carlo (MC) Dropout.

---

## 🔍 Objective

Build a robust classification model that can label political statements across six truthfulness categories:

- **True**
- **Mostly True**
- **Half True**
- **Mostly False**
- **False**
- **Pants on Fire**

---

## 🧰 Techniques Used

### 🛠 Baseline Models
- **Logistic Regression**, **Naive Bayes**, **SVM** (from prior studies)

### 🤖 Advanced Models
- **BERT** (text-only baseline)
- **Hybrid Model** (BERT + metadata like speaker & party)
- **Hybrid MC** (Hybrid + Monte Carlo Dropout)

---

## 🧪 Pipeline

1. **Data Preprocessing**
   - RoBERTa tokenization
   - One-hot encoding metadata (speaker, party)
   - Handling missing values
   - Balancing classes via oversampling

2. **Model Training**
   - Transformer-based text encoding
   - Metadata fusion
   - MC Dropout for uncertainty estimation

3. **Evaluation Metrics**
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - AUC-ROC

---

## 📊 Results

| Model          | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|----------------|----------|-----------|--------|----------|---------|
| Hybrid MC v1   | 0.272    | 0.285     | 0.272  | 0.269    | 0.646   |
| Hybrid MC v2   | 0.271    | 0.277     | 0.271  | 0.270    | 0.636   |
| Hybrid MC v3   | **0.282**| **0.291** | **0.282**| **0.281**| 0.624   |

- Hybrid MC v3 showed the best F1-score and overall performance.
- AUC-ROC peaked earlier in training, suggesting room for better calibration.

---

## 🧠 Key Insights

- Metadata significantly boosts performance when fused with text embeddings.
- MC Dropout improves generalization by modeling prediction uncertainty.
- Progressive fine-tuning avoids catastrophic forgetting and aids stability.

---

## 🚧 Challenges

- Low accuracy overall due to complex linguistic patterns.
- Inconsistent AUC-ROC across training phases.
- Limited training data for nuanced truth classification.

---

## 🔮 Future Work

- Confidence calibration (e.g., temperature scaling, Bayesian methods)
- Advanced data augmentation (e.g., paraphrasing, adversarial examples)
- Feature ablation studies to identify and prioritize impactful metadata

---

## 📂 Project Structure
```
├── data/ # Processed LIAR dataset
├── notebooks/ # Training, Hybrid, and Evaluation Notebooks
├── models/ # Saved model checkpoints
├── utils/ # Data loaders, tokenizers, and metric functions
├── results/ # Plots, classification reports
└── README.md
```

## 📚 References

- Wang, W. Y. (2017). LIAR: A Benchmark Dataset for Fake News Detection.
- Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.
- Gal & Ghahramani (2016). Dropout as a Bayesian Approximation.
- Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.

---
