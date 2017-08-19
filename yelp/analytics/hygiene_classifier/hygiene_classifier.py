import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
import numpy as np
import pandas as pd

cleaned_review_file = 'corpus/clean_reviews.txt'
training_label = 'Hygiene/training_hygiene.dat.labels'
additional = 'Hygiene/hygiene.dat.additional'

def read_data(file_name):
    with open (file_name, 'r') as f:
        text = f.readlines()
    return text

def save_file(output_file, data):
    with open ( output_file, 'w' ) as f:
        f.write('\n'.join(data))

def trained_feature_analysis(vectorizer, train_data_features):
    vocab = vectorizer.get_feature_names()
    print vocab

    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    # For each, print the vocabulary word and the number of times it 
    # appears in the training set
    for tag, count in zip(vocab, dist):
        print count, tag             

def bow():
    corpus = read_data(cleaned_review_file)
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None,
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000, 
                             min_df = 2,
                             max_df = 0.5)

    train_data_features = vectorizer.fit_transform(corpus[:546])
    train_data_features = train_data_features.toarray()

    #trained_feature_analysis(vectorizer, train_data_features)

    return train_data_features

def main():
    train = pd.read_csv(training_label, header=0)
    
    data_features = bow()

    t_size = [50,100,150,200,250,300,350]

    for i in t_size:
        train_data_features = data_features[:i, :]
        test_data_features = data_features[i:546, :]
        #forest = RandomForestClassifier(n_estimators = 100)
        logistic = LogisticRegression()
        logistic = logistic.fit(train_data_features, train['y'][:i])
        y_pred = logistic.predict(test_data_features)

        f1_score = metrics.f1_score(train['y'][i:546], y_pred)
        print 'Logistic Regression - train_size: %i f1_score: %f' % (i, f1_score)

if __name__ == '__main__':
    main()