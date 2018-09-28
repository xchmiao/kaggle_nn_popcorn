from nlp_utils import Preprocess
from utils import load_data
import logging
import pickle
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)


if __name__ == "__main__":
    train, test, unlabeled = load_data()
    i = 0
    docs = []
    prep = Preprocess()
    logging.info("Start processing the training data.")
    for review in train["review"]:
        if i % 2000 == 0:
            logging.info("Processing {}-th review.".format(i))
        docs.append(prep.clean_review_tfidf(review))
        i = i+1

    logging.info("Start processing the unlabeled data.")
    i = 0
    for review in unlabeled["review"]:
        if i % 2000 == 0:
            logging.info("Processing {}-th review.".format(i))
        docs.append(prep.clean_review_tfidf(review))
        i = i+1

    with open("docs", "wb") as fp:
        pickle.dump(docs, fp)

    fp.close()

    del docs

    logging.info("Start processing the test data.")
    i = 0
    test_docs = []
    for review in test["review"]:
        if i % 2000 == 0:
            logging.info("Processing {}-th review.".format(i))
        test_docs.append(prep.clean_review_tfidf(review))
        i = i + 1

    with open("docs", "wb") as fp:
        pickle.dump(test_docs, fp)

    fp.close()

    logging.info("Done.")
