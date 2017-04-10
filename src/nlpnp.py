#
#   Numpy-based vectorized embeddings for NLP
#
#   Modified: 8/4/17
#
# ==================================================================

from collections import Counter
from functools import lru_cache
from typing import List
import numpy as np
import nltk

def flatten(llist):
    return [elem for sublist in llist for elem in sublist]

class Numeric:
    def __init__(self, filepath: str):
        try:
            with open(filepath, 'r', encoding='utf-8') as _file:
                self.text = _file.read()
        except IOError as error:
            print(error)
        
        tokens = (nltk.word_tokenize(self.text))
        self.lexicon = {token: i for i, token in enumerate(tokens, 1)}
        self.lexicon['UNK'] = 0
        self.embeddings = None
        self.probs = None

    def lexicon(self):
        return self.lexicon

    def embed_sentence(self, sent: List[str]):
        sent_id = [self.lexicon[word] for word in sent]
        return np.array(sent_id, dtype='int')
    
    @lru_cache(maxsize=1)
    def embed(self):
        sents = nltk.sent_tokenize(self.text)
        word_sents = [nltk.word_tokenize(sent) for sent in sents]
        self.probs = dict(collections.Counter(flatten(word_sents)))
        self.probs['UNK'] = 0.0
        sum_probs = sum(self.probs.values())
        self.probs = {o: prob/sum_probs for o, prob in self.probs.items()}
        maxlen = max(map(len, word_sents))
        for wsent in word_sents:
            currlen = len(wsent)
            diff = maxlen - currlen
            if diff > 0:
                wsent += diff * ['UNK']
        a = [[self.lexicon[w] for w in j] for j in word_sents]
        b = [[self.probs[w] for w in j] for j in word_sents]
        self.embeddings = np.array(a, dtype='int')
        return self.embeddings, np.array(b, dtype='float')

