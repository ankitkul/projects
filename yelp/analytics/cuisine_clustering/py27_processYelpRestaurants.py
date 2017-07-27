import math
import json
import pickle
import random
from gensim import models
from gensim import matutils
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from nltk.tokenize import sent_tokenize
import glob
import argparse
import os
path2files="../yelp_dataset_challenge_academic_dataset/"
path2buisness=path2files+"yelp_academic_dataset_business.json"
path2reviews=path2files+"yelp_academic_dataset_review.json"


def sim_matrix():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    K_clusters = 10
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                     min_df=2, stop_words='english',
                                     use_idf=False)

    
    
    if not os.path.isdir("categories"):
        print "you need to generate the cuisines files 'categories' folder first"
        return
    
    text = []
    c_names = []
    cat_list = glob.glob ("categories/*")
    cat_size = len(cat_list)
    if cat_size < 1:
        print "you need to generate the cuisines files 'categories' folder first"
        return
    
    sample_size = min(30, cat_size)
    cat_sample = sorted( random.sample(range(cat_size), sample_size) )
    #print (cat_sample)
    count = 0
    for i, item in enumerate(cat_list):
        if i == cat_sample[count]:
            li =  item.split('/')
            cuisine_name = li[-1]
            c_names.append(cuisine_name[:-4].replace("_"," "))
            with open ( item ) as f:
                text.append(f.read().replace("\n", " "))
            count = count + 1
        
        if count >= len(cat_sample):
            print "generating cuisine matrix with:", count, "cuisines"
            break

    if len(text) < 1:
        print "the 'categories' folder does not contain any cuisines. Run this program ussing the '--cuisine' option"
    t0 = time()
    print("Extracting features from the training dataset using a sparse vectorizer")
    X = vectorizer.fit_transform(text)
    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)

    corpus = matutils.Sparse2Corpus(X,  documents_columns=False)
    lda = models.ldamodel.LdaModel(corpus, num_topics=100)

    doc_topics = lda.get_document_topics(corpus)
    cuisine_matrix = [] #similarity of topics
    # computing cosine similarity matrix
    for i, doc_a in enumerate(doc_topics):
        #print (i)
        sim_vecs = []
        for j , doc_b in enumerate(doc_topics):
            w_sum = 0
            if ( i <= j ):
                norm_a = 0
                norm_b = 0
                
                for (my_topic_b, weight_b) in doc_b:
                    norm_b = norm_b + weight_b*weight_b

                for (my_topic_a, weight_a) in doc_a:
                    norm_a = norm_a + weight_a*weight_a
                    for (my_topic_b, weight_b) in doc_b:
                        if ( my_topic_a == my_topic_b ):
                            w_sum = w_sum + weight_a*weight_b

                norm_a = math.sqrt(norm_a)
                norm_b = math.sqrt(norm_b)
                denom = (float) (norm_a * norm_b)
                if denom < 0.0001:
                    w_sum = 0
                else:
                    w_sum = w_sum/(denom)
            else:
                w_sum = cuisine_matrix[j][i]
            sim_vecs.append(w_sum)

        cuisine_matrix.append(sim_vecs)

    with open( 'cuisine_sim_matrix.csv', 'w') as f:
        for i_list in cuisine_matrix:
            s = ""
            my_max = max(i_list)
            for tt in i_list:
                s = s+str(tt/my_max) + " "
            s = s.strip()
            f.write(",".join(s.split())+"\n") #should the list be converted to m

    
    with open('cuisine_indices.txt', 'w') as f:
        f.write( "\n".join(c_names))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='This program transforms the Yelp data and saves the cuisines in the category directory. It also samples reivews from Yelp. It can also generates a cuisine similarity matrix.')
    
    parser.add_argument('--cuisine', action='store_true',
                       help='Saves a sample (10) of the cuisines to the "categories" directory. For Task 2 and 3 you will experiment with individual cuisines. This option allows you to generate a folder that contains all of the cuisines in the Yelp dataset. You can run this multiple times to generate more samples or if your machine permits you can change a sample parameter in the code.')
    parser.add_argument('--sample', action='store_true',
                       help='Sample a subset of reviews from the yelp dataset which could be useful for Task 1. This will samples upto 100,000 restaurant reviews from 10 cuisines and saves the output in "review_sample_100000.txt", it also saves their corresponding raitings in the "review_ratings_100000.txt" file. You can run this multiple times to get several different samples.')
    parser.add_argument('--matrix', action='store_true',
                       help='Generates the cuisine similarity matrix which is used for Task 2. First we apply topic modeling to a sample (30) of the cuisines in the "categories" folder and measures the cosine similarity of two cuisines from their topic weights. This might take from half-an-hour to several hours time depending on your machine. The number of topics is 20 and the default number of features is 10000.')
    parser.add_argument('--all', action='store_true',
                       help='Does all of the above.')

    
    args = parser.parse_args()        
    if args.matrix or args.all:
        print "generating cuisine matrix"
        sim_matrix()
    #main()
