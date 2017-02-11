#
# Implementation of HierSum algorithm (Hagigi, Wandervende 2009)
# Extractive summarization of corpus
# Alind Gupta 2017
#
# =========================================

import numpy as np
import nltk
import collections
import string




class MetaHierSum:

    punct = string.punctuation
    stopwords = nlkt.corpus.stopwords.words("english")

    def __init__(self, corpus: str):

        self.corpus = corpus 
        self.sents = nltk.sent_tokenize(self.corpus)    
        self.tokens = (nltk.word_tokenize(sent) for sent in self.sents)
        self.tokens = (token.lower() for token in self.tokens 
                        if token not in self.punct and token not in self.stopwords)
        self.unigram_probs = self.unigram_prob()
        self.unigram_probs_mutable = self.unigram_probs.copy()
    
    @cached_property
    def unigram_prob(self):
        """ Returns unigram probabilities as a collection.Counter """
        counter = collection.Counter(self.tokens)
        counter_length = len(counter)
        prob_sum = 0        # check may be removed in future
        for word, freq in counter:
            freq /= counter_length
            prob_sum += freq
        assert abs(prob_sum, 1.0) < 0.001, "Probability does not sum to 1"
        return sorted(counter, key=counter.get)

    
    @classmethod
    def kl_divergence(p, q):
        # if zero probs in p or q, reassign a small value and normalize p and q to 1
        def normalize_dist(x):
            x /= np.sum(x)
            return x

        if 0.0 in p:
            p[p == 0.0] = 1e-100
            p = normalize_dist(p)
        if 0.0 in q:
            q[q == 0.0] = 1e-100
            q = normalize_dist(q)



        kld = np.sum(np.multiply(p, np.log(p / q)))

        

    def sum_basic(self):
        """ SumBasic implementation """ # need a global probability update rather than this
        visited = collections.Counter()
        def sent_probs(sent: list):
            sent_prob = 0
            for word in sent:
                sent_prob += self.counter[word] ** (2 ** visited[word])
                visited.update(word)
            score = sent_prob / len(sent)
            return score
        
        # find argmax(sent_probs)
        argmax_sents = {}
        for sent in self.sents:
           if sent_probs(sent) >  


# Dirichlet parameter estimation

    
