import json
import random

import nltk
import numpy as np
from collections import Counter
import re
import pandas as pd

from gensim.models import Word2Vec
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


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


# Augment training data
def random_swap(sentence, n):
    if sentence is None:
        return

    words = sentence.split()
    if len(words) < 2:
        return

    for _ in range(n):
        position1, position2 = random.sample(range(len(words)), 2)
        words[position1], words[position2] = words[position2], words[position1]

    return ' '.join(words)


def augment_data(sentences1, sentences2, labels):
    augmented_sentences1 = []
    augmented_sentences2 = []
    augmented_labels = []
    count = 0

    for (sentence1, sentence2), label in zip(zip(sentences1, sentences2), labels):
        augmented_sentences1.append(sentence1)
        augmented_sentences2.append(sentence2)
        augmented_labels.append(label)

        if label in [0, 1]:
            augmented_text1 = random_swap(sentence1, random.randint(0, 10))
            augmented_text2 = random_swap(sentence2, random.randint(0, 10))

            if augmented_text1 is not None and augmented_text2 is not None:
                augmented_sentences1.append(augmented_text1)
                augmented_sentences2.append(augmented_text2)
                augmented_labels.append(label)

            # provide more samples to class 1
            if label == 1 and count <= 3000:
                augmented_text1 = random_swap(sentence1, random.randint(0, 5))
                augmented_text2 = random_swap(sentence2, random.randint(0, 5))

                if augmented_text1 is not None and augmented_text2 is not None:
                    augmented_sentences1.append(augmented_text1)
                    augmented_sentences2.append(augmented_text2)
                    augmented_labels.append(label)
                    count += 1

    return np.array(augmented_sentences1), np.array(augmented_sentences2), np.array(augmented_labels)


for _ in range(2):
    s1_train, s2_train, y_train = augment_data(s1_train, s2_train, y_train)

# Print out class occurrences in training dataset
label_counts = Counter(y_train)

for label, count in label_counts.items():
    print(f'Class {label}: {count} occurrences')


# Preprocess data
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()


def clean_text(sentence, lemma=False):
    sent = sentence.lower()
    sent = re.sub(r'[^\w\s]', '', sent)

    tokens = word_tokenize(sent)
    if lemma:
        tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return ' '.join(tokens)


clean_s1_train = [clean_text(sentence, lemma=True) for sentence in s1_train]
clean_s2_train = [clean_text(sentence, lemma=True) for sentence in s2_train]

clean_s1_val = [clean_text(sentence, lemma=True) for sentence in s1_val]
clean_s2_val = [clean_text(sentence, lemma=True) for sentence in s2_val]

clean_s1_test = [clean_text(sentence, lemma=True) for sentence in s1_test]
clean_s2_test = [clean_text(sentence, lemma=True) for sentence in s2_test]


def vectorize(vectorizer, sentences1, sentences2):
    sentences_vector = []
    for sentence1, sentence2 in zip(sentences1, sentences2):
        words1 = sentence1.split()
        words2 = sentence2.split()

        # Filter out-of-vocabulary words
        words1 = [word for word in words1 if word in vectorizer.wv]
        words2 = [word for word in words2 if word in vectorizer.wv]

        if words1 and words2:
            # Calculate the mean vector for each sentence
            sentence1_vector = np.mean([vectorizer.wv[word] for word in words1], axis=0)
            sentence2_vector = np.mean([vectorizer.wv[word] for word in words2], axis=0)

            # Calculate cosine similarity
            similarity = cosine_similarity([sentence1_vector], [sentence2_vector])
            sentences_vector.append(similarity[0][0])
        else:
            # Handle the case where all words are out-of-vocabulary
            sentences_vector.append(0.0)

    return np.array(sentences_vector)


# Vectorize sentences
sentences = np.concatenate((clean_s1_train, clean_s2_train), axis=0)
word2vec_vectorizer = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
word2vec_vectorizer.save('word2vec_vectorizer.bin')

x_train = vectorize(word2vec_vectorizer, clean_s1_train, clean_s2_train)
x_val = vectorize(word2vec_vectorizer, clean_s1_val, clean_s2_val)
x_test = vectorize(word2vec_vectorizer, clean_s1_test, clean_s2_test)

# Scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.fit_transform(x_val)
x_test_scaled = scaler.transform(x_test)

# Perform Grid search
grid_params = {'n_estimators': [10, 50, 100], 'criterion': ['gini', 'entropy']}
rf_model = RandomForestClassifier(n_jobs=-1)

grid_search = GridSearchCV(rf_model, grid_params, cv=5, scoring='f1_macro', verbose=1)
grid_search.fit(x_train_scaled, y_train)

print(f'Best parameters: {grid_search.best_params_}')
print(f'Best macro f1-score: {grid_search.best_score_ * 100}%')

# Score the model and print confusion matrix on the validation set
model = grid_search.best_estimator_
y_pred = model.predict(x_val_scaled)

print(classification_report(y_val, y_pred))
print('Confusion Matrix:')
metrics.confusion_matrix(y_val, y_pred)

# Test the model
y_test = model.predict(x_test_scaled)

# Write data to the CSV file
data = {'guid': guid_test, 'label': y_test}
df = pd.DataFrame(data)
df.to_csv('output1.csv', index=False)
print(f'Data has been written to output1.csv')
