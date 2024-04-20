import os
import nltk
import numpy as np
nltk.data.path.append("usr/local/share/nltk_data")

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
   
    return nltk.word_tokenize(sentence)


def stem(word):
    
    return stemmer.stem(word.lower())

#sentence = "Hi! How can I help you today?"
#tokenized_sentence = tokenize(sentence)
#print(tokenized_sentence)

def bag_of_words(tokenized_sentence, words):
    
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag