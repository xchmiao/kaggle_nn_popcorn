
# coding: utf-8

import pandas as pd
import pickle
from nlp_utils import Preprocess
import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)


train = pd.read_csv("labeledTrainData.tsv", header = 0, delimiter='\t', quoting=3)
test = pd.read_csv("testData.tsv", header = 0, delimiter='\t', quoting=3)
wv_train = pd.read_csv("unlabeledTrainData.tsv", header = 0, delimiter='\t', quoting=3)

prep = Preprocess()

sentences = []
clean_reviews = []
logging.info("Parsing sentences and generating clean text from training data.")

i = 0
for review in train["review"]:
    if i%1000 == 0:
        logging.info("Parsing {}-th review in the training data.".format(i+1))
    sentences.append(prep.review_to_sentences(review))
    clean_reviews.append(prep.review_to_wordlist(review))
    i = i+1

logging.info("Parsing sentences from unlabeled data.")
i = 0
for review in wv_train["review"]:
    if i%1000 == 0:
        logging.info("Parsing {}-th review in unlabeled data.".format(i+1))
    sentences.append(prep.review_to_sentences(review))
    i = i+1

logging.info("Generating clean text from test data.")
i = 0
for review in test["review"]:
    if i%1000 == 0:
        logging.info("Parsing {}-th review in the test data.".format(i+1))
    clean_reviews.append(prep.review_to_wordlist(review))
    i = i+1

list_dict = {"sentences": sentences,
             "clean_reviews": clean_reviews}

for filename, l in list_dict.items():
    with open(filename, "wb") as fp:
        pickle.dump(l, fp)
    fp.close()


logging.info("Done.")

