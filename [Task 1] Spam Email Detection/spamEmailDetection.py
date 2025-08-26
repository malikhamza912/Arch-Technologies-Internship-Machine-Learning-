import os
os.chdir(os.path.dirname(__file__))

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("email.csv")

#print(df.head())
#print(df.info())
#print(df.isnull().sum())

x = df["Message"]
y = df["Category"]

label_mapping = {"spam": 1, "ham": 0}
y = df["Category"].map(label_mapping)

#print(y.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

#print(len(x_train), len(x_test))
#print(len(y_train), len(y_test))

vectorizer = CountVectorizer()
x_trainVectors = vectorizer.fit_transform(x_train)
x_testVectors = vectorizer.transform(x_test)

model = MultinomialNB()
model.fit(x_trainVectors, y_train)

y_pred = model.predict(x_testVectors)

from sklearn.metrics import accuracy_score, confusion_matrix
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))