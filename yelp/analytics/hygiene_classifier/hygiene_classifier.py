from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as metrics
from sklearn import svm
import numpy as np
import pandas as pd
import csv
from sklearn.cross_validation import train_test_split

unigram_review = 'corpus/unigram_reviews.txt'
mixed_corpus = 'corpus/mixed_corpus.txt'
training_label = 'Hygiene/training_hygiene.dat.labels'
additional = 'Hygiene/hygiene.dat.additional'

def read_data(file_name):
    with open (file_name, 'r') as f:
        text = f.readlines()
    return text

def save_csv(output_file, data, header_list):
    with open(output_file,'wb') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(header_list)
        for row in data:
            csv_out.writerow(row)

def trained_feature_analysis(vectorizer, train_data_features):
    vocab = vectorizer.get_feature_names()
    print vocab

    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    for tag, count in zip(vocab, dist):
        print count, tag

def custom_tokens(text):
    tokens = []
    for item in text.split(','):
        tokens.append(item.replace(' ', '_'))
    return tokens

def bag_of_words(tfidf, file):
    corpus = read_data(file)

    tokenizer_func = None
    if file == mixed_corpus:
        tokenizer_func = custom_tokens

    if tfidf:
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=5000,
                             min_df=2, stop_words = None,
                             use_idf=True,
                             tokenizer = tokenizer_func)
    else:
        vectorizer = CountVectorizer(analyzer = "word", tokenizer = tokenizer_func,
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

    tfidf = False
    data_features = bag_of_words(tfidf, mixed_corpus)

    t_size = [0.4,0.5,0.6,0.7,0.8]

    names = ['Logistic Regression', 'SVM', 'Linear SVM',
                'Decision Tree', 'Random Forest', 'Naive Bayes']
    classifiers = [LogisticRegression(),
                    svm.SVC(),
                    svm.LinearSVC(),
                    DecisionTreeClassifier(max_depth=5),
                    RandomForestClassifier(n_estimators = 100),
                    GaussianNB()]

    f1_score_train_size = []
    for i in t_size:

        X_train, X_test, y_train, y_test = train_test_split(data_features[:546],
                                            train['y'][:546],
                                            train_size=i,
                                            random_state=44)

        for name, clf in zip(names, classifiers):

            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            #prob = clf.predict_proba(test_data_features)

            f1_score = metrics.f1_score(y_test, y_pred, average = 'micro')

            f1_score_train_size.append([name, str(i * 100), str(f1_score)])

            print ','.join([name, str(i * 100), str(f1_score)])

    if tfidf:
        tf_text = 'tfidf' #lgtm [py/unreachable-statement]
    else:
        tf_text = 'tf'
    
    csv_header = ['model_name','train_labe_sample','f1_score']
    save_csv('output/f1_score_train_size_%s.txt' % tf_text,
                f1_score_train_size,
                csv_header)

if __name__ == '__main__':
    main()