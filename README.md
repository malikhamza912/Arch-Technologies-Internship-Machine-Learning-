# 🧠 Machine Learning Projects

This repository contains two Machine Learning tasks:

- ✅ **Spam Email Detection** – Classify messages as Spam or Ham using Naive Bayes  
- ✅ **MNIST Digit Recognition** – Recognize handwritten digits using Neural Networks  

---

## 📌 Project Overview

### **1. Spam Email Detection**
**Goal:** Detect whether an email is spam or not using text classification.

- **Algorithm:** Multinomial Naive Bayes  
- **Feature Extraction:** `CountVectorizer`  
- **Accuracy Achieved:** **97.9%**

#### ✅ Key Steps:
- Load & clean dataset  
- Encode labels (`spam` → 1, `ham` → 0)  
- Convert text to numeric features  
- Train and evaluate model  

#### 🔍 Sample Confusion Matrix:
[[946 7]
[ 16 146]]


---

### **2. MNIST Digit Recognition**
**Goal:** Classify handwritten digits (0–9) from the MNIST dataset.

- **Model:** Neural Network (Dense layers)  
- **Accuracy Achieved:** **97%**

#### ✅ Model Architecture:
Flatten(28x28) → Dense(128, ReLU) → Dense(10, Softmax)


#### ✅ Key Steps:
- Normalize image data  
- Build & compile the model  
- Train and evaluate  

---

## 🛠 Tech Stack
- **Language:** Python 3.10+  
- **Libraries:**  
  - `numpy`, `pandas`  
  - `scikit-learn`  
  - `tensorflow / keras`  

---

## ▶ How to Run

```bash
# Clone the repository
git clone https://github.com/malikhamza912/Arch-Technologies-Internship-Machine-Learning-
cd Machine-Learning-Projects

# Install dependencies
pip install numpy pandas scikit-learn tensorflow

# Run Spam Detection
python spamEmailDetection.py

# Run MNIST Digit Recognition
python digitRecognition.py
