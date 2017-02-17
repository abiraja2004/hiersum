#
#   Implementation of BaseSum
#   Extractive summarization of articles
#
#   Alind Gupta (2017)
#   Comments: works well
#
# ==========================================================


import io
import string
from collections import Counter
from typing import Dict, List, Tuple
import nltk

# type aliases
Probability = float
Token = str
Sentence = str


class BaseSum:
    """
    Extractive summarization of sentences based on unigram
    frequencies in a sentence.

    Metric described in: #ref
    """

    # invalid tokens, includes punctuation and stopwords
    invalid = set(list(string.punctuation) +
                  nltk.corpus.stopwords.words('english'))

    def __init__(self, filepath: str):
        try:
            with io.open(filepath, 'rt', encoding='utf-8') as _file:
                self.text = _file.read()
        except IOError as error:
            print('Could not read file:', filepath)
            print(error)

    @staticmethod
    def is_numeric(obj: str) -> bool:
        """ Returns True if argument can be coerced to type 'float' """
        try:
            float(obj)
            return True
        except ValueError:
            return False

    @staticmethod
    def normalize_probs(probs: Dict[Token, Probability]):
        """ Normalize probabilities to sum to 1 """
        sum_probs = sum(probs.values())
        return {o: prob/sum_probs for o, prob in probs.items()}

    @staticmethod   # pure
    def unigram_probs(corpus: List[Token]) -> Dict[Token, Probability]:
        """ Returns dict containing tokens and their normalized frequencies """
        counter = Counter(corpus)
        return BaseSum.normalize_probs(counter)

    @staticmethod   # pure
    def sent_score(token_probs: Dict[Token, Probability],
                   sent: List[Token],
                   normalize_lengths=False) -> float:
        """ Calculate score of a sentence by summing probability of token """
        score = sum([token_probs[token] for token in sent])
        return score if normalize_lengths else score/len(sent)

    @staticmethod   # pure
    def handle_invalid_tokens(unigram_probs: Dict[Token, Probability]):
        """ Probability of invalid tokens is set to 0 """
        new_probs = {token: 0.0 if token in
                            BaseSum.invalid or BaseSum.is_numeric(token)
                            else prob
                     for token, prob in unigram_probs.items()}
        return BaseSum.normalize_probs(new_probs)

    def text_handler(self) -> Tuple[List[List[Token]], List[Token]]:
        """ Handles text """
        sents = iter(nltk.sent_tokenize(self.text))
        sents_nested = [nltk.word_tokenize(sent)
                        for sent in sents]
        tokens = [word for sentence in sents_nested
                  for word in sentence]
        return (sents_nested, tokens)

    def summarize(self, num_sents: int) -> List[Sentence]:
        """ Main method """
        sents, tokens = self.text_handler()
        if not sents or not tokens:
            raise IOError('Empty input')

        # generate token probability dict
        valid_unigram_probs = BaseSum.handle_invalid_tokens(
            BaseSum.unigram_probs(tokens))

        # helper function   # impure
        def update_probs(unigram_probs, counter: Dict[Token, int]):
            """ Scale probabilities according to counter """
            for token in counter:
                unigram_probs[token] **= (2 * counter[token])
            return unigram_probs

        # initialize empty list to hold summary sentences
        summary = []

        # loop to extract summary sentences
        while len(summary) < num_sents:
            sent_scores = {BaseSum.sent_score(valid_unigram_probs, sent): sent
                           for sent in sents}
            top_sent = sent_scores[max(sent_scores.keys())]

            # __DEBUG__
            print(max(sent_scores.keys()))

            elem_top_sent = sents.index(top_sent)
            counter = Counter(top_sent)
            valid_unigram_probs = update_probs(valid_unigram_probs, counter)
            summary.append((' '.join(top_sent)))

            # __DEBUG__
            print(summary[-1])

            del sents[elem_top_sent]

        # print(summary)

        # print average sentence length
        temp = list(map(len, summary))
        print('Average summary sentence length: ', sum(temp) / len(temp))

        return summary
