import os
import re
from nltk.corpus import stopwords
import numpy as np

review_file = 'Hygiene/hygiene.dat'

def read_data(file_name):
    with open (file_name, 'r') as f:
        text = f.readlines()
    return text

def save_file(output_file, data):
    with open ( output_file, 'w' ) as f:
        f.write('\n'.join(data))     

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review) 
    #
    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 3. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 4. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 5. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))
    

def main():
    reviews = read_data(review_file)

    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []
    for review in reviews:
        clean_train_reviews.append(review_to_words(review))
    save_file('corpus/unigram_reviews.txt', clean_train_reviews)    

if __name__ == '__main__':
    main()