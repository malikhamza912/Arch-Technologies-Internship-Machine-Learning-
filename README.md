This repository contains two core Machine Learning tasks implemented in Python:

✅ Spam Email Detection – Classify messages as Spam or Ham using Naive Bayes
✅ MNIST Digit Recognition – Recognize handwritten digits using Neural Networks

📌 Projects Overview
1. Spam Email Detection

Goal: Detect whether an email is spam or not using text classification.

Algorithm: Multinomial Naive Bayes

Feature Extraction: CountVectorizer

Accuracy Achieved: ~97.9%

Key Steps:
✔ Load & clean dataset
✔ Map labels (spam=1, ham=0)
✔ Convert text to feature vectors
✔ Train & evaluate model

Sample Confusion Matrix:

[[946   7]
 [ 16 146]]

2. MNIST Digit Recognition

Goal: Classify handwritten digits (0–9) from the MNIST dataset.

Model: Neural Network (Dense layers)

Accuracy Achieved: ~97%

Architecture:

Flatten(28x28) → Dense(128, ReLU) → Dense(10, Softmax)


Key Steps:
✔ Normalize image data
✔ Build & compile the model
✔ Train and evaluate

🛠 Technologies Used

Python 3.10+

Libraries:

pandas, numpy

scikit-learn

tensorflow / keras

▶ How to Run
# Clone the repository
git clone https://github.com/your-username/Machine-Learning-Projects.git
cd Machine-Learning-Projects

# Install dependencies
pip install -r requirements.txt

# Run Spam Detection
python spam_email_detection.py

# Run MNIST Digit Recognition
python mnist_digit_recognition.py

📈 Future Enhancements

Implement TF-IDF in spam detection

Add CNN for MNIST to achieve 99%+ accuracy

Deploy models via Flask / FastAPI
