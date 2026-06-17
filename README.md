# 📱 SMS Spam Detection

A machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing and a Linear Support Vector Classifier.

**Final Model Accuracy: 97.76%**

---

## 📌 Project Overview

SMS spam remains a significant nuisance and security concern. This project builds an end-to-end text classification pipeline — from raw SMS text to a tuned ML model — capable of accurately detecting spam messages.

---

## 🗂️ Dataset

- **File:** `spam_cleaned.csv`
- **Size:** 5,572 SMS messages
- **Columns:** `message` (text), `target` (ham/spam label)
- **Source:** [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## 🔄 Project Workflow

```
Raw SMS Text
     │
     ▼
Text Preprocessing
  ├── Remove numbers & special characters
  ├── Convert to lowercase
  ├── Remove stopwords
  └── Apply stemming (PorterStemmer)
     │
     ▼
Feature Extraction (CountVectorizer / Bag of Words)
     │
     ▼
Model Comparison (MultinomialNB, GaussianNB, SVC)
     │
     ▼
Best Model: LinearSVC
     │
     ▼
Hyperparameter Tuning (GridSearchCV, C=[0.01, 0.1, 1, 10])
     │
     ▼
Final Evaluation (Accuracy, Confusion Matrix, Classification Report)
```

---

## 🧹 Text Preprocessing Steps

| Step | Description |
|------|-------------|
| Remove special characters | Strip numbers, punctuation, symbols using regex |
| Lowercase conversion | Standardize all text to lowercase |
| Stopword removal | Remove common words (is, the, a, etc.) using NLTK |
| Stemming | Reduce words to root form using PorterStemmer |

---

## 🤖 Models Compared

| Model | Accuracy |
|-------|----------|
| Gaussian Naive Bayes | ~87% |
| Multinomial Naive Bayes | ~98% |
| SVC | ~98% |
| **LinearSVC (Tuned)** | **97.76%** |

LinearSVC was selected as the final model due to its strong performance on high-dimensional sparse text data.

---

## ⚙️ Hyperparameter Tuning

Used **GridSearchCV** with 5-fold cross-validation to find the best regularization parameter `C` for LinearSVC.

```python
params = {'C': [0.01, 0.1, 1, 10]}
svc_cv = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', n_jobs=-1)
```

---

## 📊 Final Results

```
Accuracy: 97.76%

Classification Report:
              precision    recall  f1-score
Ham (0)          0.98       0.99      0.99
Spam (1)         0.97       0.93      0.95
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Libraries:** pandas, numpy, nltk, scikit-learn, matplotlib
- **Notebook:** Jupyter Notebook

---

## 📁 Project Structure

```
spam-detection/
│
├── Spam_Detection_project.ipynb   # Main notebook
├── spam_cleaned.csv               # Dataset
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ayush-kumar-jha953/spam-detection.git
   cd spam-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data**
   ```python
   import nltk
   nltk.download('stopwords')
   ```

4. **Run the notebook**
   ```bash
   jupyter notebook Spam_Detection_project.ipynb
   ```

---

## 📈 Key Learnings

- Text preprocessing significantly impacts model performance
- Bag of Words with CountVectorizer is effective for SMS classification
- LinearSVC handles high-dimensional sparse text data better than kernel-based SVM
- GridSearchCV with cross-validation prevents overfitting during tuning

---

## 👤 Author

**Ayush Kumar Jha**  
[![GitHub](https://img.shields.io/badge/GitHub-Ayush--kumar--jha953-black?logo=github)](https://github.com/Ayush-kumar-jha953)
