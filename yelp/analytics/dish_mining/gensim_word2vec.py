from gensim import models
import os
from collections import Counter

class MySentences_Dir(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
              yield line.split(',')

class MySentences(object):
    def __init__(self, fname, phrase):
        self.filename = fname
        self.do_phrase = phrase
 
    def __iter__(self):
      for line in open(self.filename):
        if self.do_phrase:
          yield line.split(',')
        else:
          yield line.split(' ')              


'''
dir_name = '/Users/ankit/Documents/Online Courses/Data mining/Capstone/task3/word2vec/data'
fname = 'data/corpus_indian.txt'
sentences = MySentences(fname, True) # a memory-friendly iterator
'''

# Line sentence
sentences = models.word2vec.LineSentence('data/Indian.txt')

model = models.Word2Vec(sentences, hs=1, negative=0, min_count=5, size=200, workers =4)
model.save('model-raw-word-with-score.bin')

counter = Counter()
for key in model.wv.vocab.keys():
  if len(key.split(' '))>1:
    counter[key] += model.wv.vocab[key].count

for word,cnt in counter.most_common(20):
  print word.encode("utf-8")
#clear the memory
del model
