# -*- coding: utf-8 -*-

""" basesum.py

Basic summarizer, which will calculate
unigram frequencies in the corpus and extract
sentences which have a high probability of representing the corpus
based on unigram makeup (excluding stop words etc. of course).

Requires Python 3.6 or higher.

"""
import string
from collections import Counter
from typing import Dict, List, Tuple, IO
import nltk

# type aliases
Probability = float
Token = str
Sentence = str

# invalid tokens, includes punctuation and stopwords
invalid = set(list(string.punctuation) +
              nltk.corpus.stopwords.words('english'))


def is_numeric(obj: str) -> bool:
    """ Returns True if argument can be coerced to type 'float' """
    try:
        float(obj)
        return True
    except ValueError:
        return False

    
class BaseSum:
    """ Extractive summarization of sentences based on unigram
    frequencies in a sentence.

    Metric described in: #ref
    """
    def __init__(self, filepath: str) -> IO[str]:
        try:
            with open(filepath, 'rt', encoding='utf-8') as file_handle:
                self.text = file_handle.read()
        except IOError:
            raise
        
    @staticmethod
    def normalize_probs(probs: Dict[Token, Probability]) \
        -> Dict[Token, Probability]:
        """ Normalize probabilities to sum to 1 """
        sum_probs = sum(probs.values())
        return {o: prob/sum_probs for o, prob in probs.items()}

    @staticmethod
    def unigram_probs(corpus: List[Token]) -> Dict[Token, Probability]:
        """ Returns dict containing tokens and their normalized frequencies """
        counter = Counter(corpus)
        return BaseSum.normalize_probs(counter)

    @staticmethod
    def sent_score(token_probs: Dict[Token, Probability],
                   sent: List[Token],
                   normalize_lengths=False) -> float:
        """ Calculate score of a sentence by summing probability of token """
        score = sum([token_probs[token] for token in sent])
        return score if normalize_lengths else score/len(sent)

    @staticmethod
    def handle_invalid_tokens(unigram_probs: Dict[Token, Probability]):
        """ Probability of invalid tokens is set to 0 """
        new_probs = {token: 0.0 if token in
                            invalid or is_numeric(token)
                            else prob
                     for token, prob in unigram_probs.items()}
        return BaseSum.normalize_probs(new_probs)

    def text_handler(self) -> Tuple[List[List[Token]], List[Token]]:
        """ Tokenize the corpus into sentences, and sentences into unigrams
        and return a tuple containing sentences and unigrams.

        """
        sents = iter(nltk.sent_tokenize(self.text))
        sents_nested = [nltk.word_tokenize(sent)
                        for sent in sents]
        tokens = [word for sentence in sents_nested
                  for word in sentence]
        return (sents_nested, tokens)

    def summarize(self, num_sents: int) -> List[Sentence]:
        """ Generate an extractive summary.
        
        Parameters
        ----------
        num_sents: int
            Number of sentences of summary to output

        Returns
        -------
        List[str]
            Extracted summary of length ``num_sents``

        """
        sents, tokens = self.text_handler()
        if not sents or not tokens:
            raise IOError('Empty input')

        # generate token probability dict
        valid_unigram_probs = BaseSum.handle_invalid_tokens(
            BaseSum.unigram_probs(tokens))

        def update_probs(unigram_probs, counter: Dict[Token, int]):
            """ Scale probabilities according to counter """
            for token in counter:
                unigram_probs[token] **= (2 * counter[token])
            return unigram_probs

        summary = []
        while len(summary) < num_sents:
            sent_scores = {BaseSum.sent_score(valid_unigram_probs, sent): sent
                           for sent in sents}
            top_sent = sent_scores[max(sent_scores.keys())]
            elem_top_sent = sents.index(top_sent)
            counter = Counter(top_sent)
            valid_unigram_probs = update_probs(valid_unigram_probs, counter)
            summary.append((' '.join(top_sent)))
            del sents[elem_top_sent]

        # print average sentence length
        temp = list(map(len, summary))
        print('Average summary sentence length: ', sum(temp) / len(temp))
        return summary
