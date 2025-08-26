🧠 Machine Learning Projects

This repository contains two fundamental Machine Learning projects implemented in Python:

Spam Email Detection – Classify emails as Spam or Ham using Naive Bayes.

MNIST Digit Recognition – Recognize handwritten digits (0-9) using Neural Networks.

📂 Project Structure
📦 Arch-Technologies-Internship-Machine-Learning
├── 📁 [Task 1] Spam Email Detection
│   ├── spamEmailDetection.py
│   ├── email.csv
|
├── 📁 [Task 2] 
│   ├── digitRecognition.py
│   
└── README.md

✅ 1. Spam Email Detection

Goal: Classify text messages as spam or ham.

Dataset

CSV file: email.csv

Columns:

Category → spam/ham

Message → email text

Steps

Preprocess labels (spam=1, ham=0).

Split data into train/test.

Convert text to feature vectors using:

CountVectorizer

(Optionally TF-IDF)

Train a Multinomial Naive Bayes model.

Evaluate using accuracy and confusion matrix.

Sample Output
Accuracy: 97.9%
Confusion Matrix:
[[946   7]
 [ 16 146]]

Code Snippet
vectorizer = CountVectorizer()
x_trainVectors = vectorizer.fit_transform(x_train)
x_testVectors = vectorizer.transform(x_test)

model = MultinomialNB()
model.fit(x_trainVectors, y_train)

✅ 2. MNIST Digit Recognition

Goal: Classify handwritten digits (0–9) from the MNIST dataset.

Dataset

Built-in Keras MNIST dataset.

60,000 training images, 10,000 test images.

Model

Neural Network (Dense):

Flatten(28×28) → Dense(128, relu) → Dense(10, softmax)

(Optional) Improve with CNN for better accuracy.

Steps

Normalize pixel values (0–1).

Build & compile model:

model = keras.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])


Train & evaluate:

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
test_loss, test_acc = model.evaluate(x_test, y_test)


Achieved Accuracy: ~97%.

🛠 Tools & Libraries

Python 3.10+

pandas, numpy

scikit-learn

tensorflow / keras

matplotlib (optional for visualization)

▶️ How to Run

Clone the repository:

git clone https://github.com/your-username/Machine-Learning-Projects.git


Install dependencies:

pip install -r requirements.txt


Run each script:

python spamEmailDetection.py
python mnistDigitRecognition.py

📈 Future Enhancements

Add TF-IDF for Spam Detection.

Implement CNN for MNIST for ~99% accuracy.

Deploy models with Flask / FastAPI.
