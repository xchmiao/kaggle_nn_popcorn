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

    def html_to_text(self, review):
        review_text = BeautifulSoup(review, "html").get_text()
        if len(review_text) < 1:
            review_text = review
        return review_text

    def remove_non_letters(self, text):
        return re.sub("[^a-zA-Z]", " ", review_text)

    def remove_sw(self, word_list):
        '''
        Remove stop words from the input word list
        :param word_list: list of strings
        :return: list of strings
        '''
        stops = set(stopwords.words("english"))
        words = filter(lambda x: x not in stops, word_list)
        return words


    def review_to_wordlist(self, review, remove_stopwords = True):
        '''
        Clean review text and transform one review to a list of words (no punctuation, no number).

        :param review: string
        :param remove_stopwords: boolean, whether or not to remove stop words from review text
        :return: list of strings
        '''

        review_text = self.html_to_text(review)
        review_text = self.remove_non_letters(review_text) # remove non-letters

        words = review_text.lower().split()

        if remove_stopwords:
            words = self.remove_sw(words)

        return words

    def review_to_sentences(self, review, remove_stopwords = True):
        '''
        Tokenize review into tokenized sentences for Word2Vec trainning.

        :param review: string
        :param remove_stopwords: boolean
        :return: list of strings
        '''

        review_text = self.html_to_text(review)
        raw_sentences = nltk.sent_tokenize(review_text.strip())

        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(self.review_to_wordlist(raw_sentence, remove_stopwords))

        return sentences

    def clean_review_tfidf(self, review, remove_stopwords = True):
        '''
        Clean the review text by converting it to html, tokenizing to sentences, removing non-letters,
        and lemmatization, and make it ready to extract TF-IDF features.

        :param review: string
        :param remove_stopwords: boolean
        :return: doc: list of lists, cleaned sentences in word-tokens
        '''

        review_text = self.html_to_text(review)
        sentences = nltk.tokenize.sent_tokenize(review_text)

        sentences = list(map(self.remove_non_letters, sentences)) #remove non-letter, but keep orginal spaces

        sentences = list(map(lambda s: s.lower().split(), sentences))

        sentences_tagged = list(map(lambda s: nltk.pos_tag(s), sentences))

        if remove_stopwords:
            stops = set(stopwords.words("english"))
            sentences_tagged = list(map(lambda l: list(filter(lambda x: x[0] not in stops, l)), sentences_tagged))

        if remove_numbers:
            sentences_tagged = list(map(lambda l: list(filter(lambda x: x[0].isalpha(), l)), sentences_tagged))

        # lemmatization
        tag_map = defaultdict(lambda: wordnet.NOUN)
        tag_map["J"] = wordnet.ADJ
        tag_map["V"] = wordnet.VERB
        tag_map["R"] = wordnet.ADV

        lemma = WordNetLemmatizer()

        doc = []
        for sent in sentences_tagged:
            s = list(map(lambda (word, tag): lemma.lemmatize(word, pos=tag_map[tag[0]]), sent))
            doc.append(" ".join(s))

        return doc