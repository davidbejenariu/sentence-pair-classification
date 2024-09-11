import json
import numpy as np
from collections import Counter
import re
import pandas as pd

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight


def parse_json(data, train=False):
    s1 = np.array([entry['sentence1'] for entry in data])
    s2 = np.array([entry['sentence2'] for entry in data])
    guids = np.array([entry['guid'] for entry in data])

    if train:
        labels = np.array([entry['label'] for entry in data])
        return s1, s2, labels, guids

    return s1, s2, guids


# Parse the given dataset
with open('data/train.json', 'r') as input_file:
    train_data = json.load(input_file)

with open('data/validation.json', 'r') as input_file:
    val_data = json.load(input_file)

with open('data/test.json', 'r') as input_file:
    test_data = json.load(input_file)

s1_train, s2_train, y_train, guid_train = parse_json(train_data, train=True)
s1_val, s2_val, y_val, guid_val = parse_json(val_data, train=True)
s1_test, s2_test, guid_test = parse_json(test_data)

# Print out class occurrences in training dataset
label_counts = Counter(y_train)

for label, count in label_counts.items():
    print(f'Class {label}: {count} occurrences')


# Preprocess data
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def tokenize(text):
    return text.split(' ')


# Vectorize data
vectorizer = TfidfVectorizer(preprocessor=preprocess, tokenizer=tokenize)

train_data = [sentence1 + ' ' + sentence2 for sentence1, sentence2 in zip(s1_train, s2_train)]
x_train = vectorizer.fit_transform(train_data)

val_data = [sentence1 + ' ' + sentence2 for sentence1, sentence2 in zip(s1_val, s2_val)]
x_val = vectorizer.transform(val_data)

test_data = [sentence1 + ' ' + sentence2 for sentence1, sentence2 in zip(s1_test, s2_test)]
x_test = vectorizer.transform(test_data)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=[0, 1, 2, 3], y=y_train)

# Perform Grid search
grid_params = {'C': [0.1, 1, 10], 'gamma': ['auto', 'scale'], 'kernel': ['linear', 'rbf']}
svc_model = SVC(class_weight=class_weights)

grid_search = GridSearchCV(svc_model, grid_params, cv=5, scoring='f1_macro', verbose=1)
grid_search.fit(x_train, y_train)

print(f'Best parameters: {grid_search.best_params_}')
print(f'Best macro f1-score: {grid_search.best_score_ * 100}%')

# Score the model and print confusion matrix on the validation set
model = grid_search.best_estimator_
y_pred = model.predict(x_val)

print(classification_report(y_val, y_pred))
print('Confusion Matrix:')
metrics.confusion_matrix(y_val, y_pred)

# Test the model
y_test = model.predict(x_test)

# Write data to the CSV file
data = {'guid': guid_test, 'label': y_test}
df = pd.DataFrame(data)
df.to_csv('output2.csv', index=False)
print(f'Data has been written to output2.csv')
