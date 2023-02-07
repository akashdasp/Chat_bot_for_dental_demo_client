import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()
def tokenize(setence):
    return nltk.word_tokenize(setence)
def stem(word):
    return stemmer.stem(word.lower())
def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence=[stem(w) for w in tokenized_sentence]
    bag=np.zeros(len(all_words),dtype=np.float32)
    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx]=1.0
    return bag

# sentece=["hello","how","are","you"]
# words=['hi','how','are','you','buy','thank','you']
# bag=bag_of_words(sentece,words)
# print(bag)



# words=["Organize","Organ","Organization","Organizing"]
# stemmed_words=[stem(word) for word in words]
