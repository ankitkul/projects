from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn import svm
import numpy as np
import pandas as pd
import csv
import ast
from sklearn.cross_validation import train_test_split

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
    return text.split(',')

def bow():
    corpus = []
    other_features = []
    with open(additional, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            tags = []
            for item in ast.literal_eval(row[0]):
                item = item.replace(' ', '_')
                tags.append(item)
            tags.append(row[1][2:])
            corpus.append(','.join(tags))
            other_features.append( [int(row[2])*1.0/139, float(row[3])/5] )

    other_features = np.array(other_features[:546])

    vectorizer = TfidfVectorizer(max_df=0.5, max_features=5000,
                             min_df=2, stop_words = None,
                             use_idf=True, tokenizer = custom_tokens)

    train_data_features = vectorizer.fit_transform(corpus[:546])
    train_data_features = train_data_features.toarray()

    trained_feature_analysis(vectorizer, train_data_features)

    combined_features = np.hstack((train_data_features, other_features))

    return combined_features

def main():
    train = pd.read_csv(training_label, header=0)

    data_features = bow()

    t_size = [0.6,0.7,0.8]

    names = ['Logistic Regression', 'Linear SVM']
    classifiers = [LogisticRegression(),
                    svm.LinearSVC()]

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

    csv_header = ['model_name','train_labe_sample','f1_score']
    save_csv('output/f1_score_additional_factors.txt',
                f1_score_train_size,
                csv_header)

if __name__ == '__main__':
    main()