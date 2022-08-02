import nltk
import numpy as np
#nltk.download('punkt')     #For first time use
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence): #split into words
    return nltk.word_tokenize(sentence)

def stem(word): #get the root word
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):#convert words to integers
    sentence_words = [stem(word) for word in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype = np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1.0

    return bag

         
    
