#
#   Numpy-based manipulations for Hiersum
#   Data utilities
#
#   
#
# ========================================

from collections import Counter
from typing import List
import numpy as np
import nltk


def flatten(x):
    return [i for j in x for i in j]


class NumpyIO:
    """ 
    """

    def __init__(self, filepath: str):
        """ """
        try:
            with open(filepath, 'r', encoding='utf-8') as _file:
                text = _file.read()
        except IOError as error:
            print(error)

        self.sents = nltk.sent_tokenize(text)
        self.tokens = [nltk.word_tokenize(sent) for sent in self.sents]

        self.lexicon = {token: _id for _id, token in enumerate(
            set(flatten(self.tokens)), 1)}
        self.lexicon['PAD'] = 0


    def embed_corpus(self):
       """ """

       # find length of longest sentences in corpus
       max_length = max(map(len, self.tokens))

       # insert 'PAD' tokens to ends to pad to max_length
       sents = self.sents
       sents = [[self.lexicon
       for sent in sents:
           sent_length = len(sent)
           diff = max_length - sent_length
           if diff > 0:
               sent += diff * ['PAD']
       return np.asarray(sents, dtype='int')



    def calculate_probs(self):
        """ """

        # counter to calculate token frequency
        counter = Counter((token for tokens_ in self.tokens
            for token in tokens_))
        sum_freqs = sum(counter.values())
        self.probs = {token: freq/sum_freqs for token, freq in counter.items()}
        prob_sents = [[self.probs[token] for token in sublist] for sublist in self.tokens]
        return np.asarray(prob_sents, dtype='int')
