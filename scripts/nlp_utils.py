import nltk
import pandas as pd 
from bs4 import BeautifulSoup
import re 
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import pickle
from collections import defaultdict

class Preprocess(object):
    def __init__(self):
        pass

    def review_to_wordlist(self, review, remove_stopwords = True, remove_numbers = False):
        '''

        :param review:
        :param remove_stopwords:
        :param remove_numbers:
        :return:
        '''

        # 1. Remove Html
        review_text = BeautifulSoup(review, "html").get_text()
        if len(review_text) < 1:
            review_text = review
        review_text = re.sub("[^a-zA-Z]", " ", review_text) # remove non-letters
        words = review_text.lower().split()

        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = filter(lambda x: x not in stops, words)

        if remove_numbers:
            words = filter(lambda x: x.isalpha(), words)

        return words

    def review_to_sentences(self, review, remove_stopwords = True, remove_numbers = False):

        review_text = BeautifulSoup(review, "html").get_text()
        if len(review_text) < 1:
            review_text = review
        raw_sentences = nltk.sent_tokenize(review_text.strip())

        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(self.review_to_wordlist(raw_sentence, remove_stopwords, remove_numbers))

        return sentences

    def clean_review_tfidf(self, remove_stopwords = True, remove_numbers = True):
        '''

        :param remove_stopwords:
        :param remove_numbers:
        :return:
        '''

        review_text = BeautifulSoup(review, "html").get_text()
        if len(review_text) < 1:
            review_text = review
        sentences = nltk.tokenize.sent_tokenize(review_text)

        sentences = map(lambda s: re.sub("[^a-zA-Z\s]", "", s), sentences) #remove non-letter, but keep orginal spaces

        sentences = map(lambda s: s.lower().split(), sentences)

        sentences_tagged = map(lambda s: nltk.pos_tag(s), sentences)

        if remove_stopwords:
            stops = set(stopwords.words("english"))
            sentences_tagged = map(lambda l: filter(lambda x: x[0] not in stops, l), sentences_tagged)

        if remove_numbers:
            sentences_tagged = map(lambda l: filter(lambda x: x[0].isalpha(), l), sentences_tagged)

        # lemmatization
        tag_map = defaultdict(lambda: wordnet.NOUN)
        tag_map["J"] = wordnet.ADJ
        tag_map["V"] = wordnet.VERB
        tag_map["R"] = wordnet.ADV

        lemma = WordNetLemmatizer()

        doc = []
        for sent in sentences_tagged:
            s = map(lambda (word, tag): lemma.lemmatize(word, pos=tag_map[tag[0]]), sent)
            doc.append(" ".join(s))

        return doc