# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:27:03 2015
"""

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_list = stopwords.words('arabic')


import codecs

import numpy as np
import pandas as pd
import re


from random import shuffle


class AraTweetReversed:
    def __init__(self):
        self.REVIEWS_PATH = "../data/"
        self.DELETED_REVIEWS_FILE = "deleted_reviews.tsv"
        self.CLEAN_REVIEWS_FILE = "Tweets.txt"

    def read_review_file(self, file_name):

        reviews = codecs.open(file_name, 'r', 'utf-8').readlines()
        # regex_empty = u"#|!|\.|\"|؟|:|\?|-|=|…|\||ö|\^|>|<|\[|]|^\s+|\s+$|[ ]{2,}"
        # regex_space = u"_|\)|\(|\,|،|[0-9]"
        regex = u'[^\w\t]|_|[0-9]'
        # remove comment lines and newlines
        reviews = [re.sub(regex, ' ', r, flags=re.UNICODE) for r in reviews]

        # reviews = [re.sub((regex_empty), "", r) for r in reviews]
        # reviews = [re.sub((regex_space), " ", r) for r in reviews]

        # reviews = [re.sub((""), "", r) for r in reviews]
        # parse
        rating = list()
        body = list()
        vocab = {}
        index = 0
        for review in reviews:
            # split by <tab>
            parts = review.split(u"\t")
            comment = parts[0].strip()
            sentiment = parts[1].strip()
            # if(sentiment == "OBJ" or sentiment == "NEG"):
            if (sentiment == "OBJ"):

                continue
            for word in comment.split(" "):
                if(not vocab.has_key(word)):
                    vocab[word] = index
                    index = index + 1

            body.append(comment)
            rating.append(sentiment)
        print len(vocab)
        return (body, rating, vocab)

    def read_clean_reviews(self):
        return self.read_review_file(self.REVIEWS_PATH +
                                     self.CLEAN_REVIEWS_FILE)

    def reverse_dataset(self,body,rating,vocab):
        print
        labels = []
        reversed_reviews = []
        for label in rating:
            if(label == "POS"):
                labels.append("1")
            elif (label == "NEG"):
                labels.append("3")
            elif (label == "NEUTRAL"):
                labels.append("0")
        for review in body:
            line = []
            for word in review.split(" "):
                line.append(vocab[word])
            reversed_reviews.append(line)
        reversed_vocab = {v: k for k, v in vocab.iteritems()}
        return reversed_reviews,reversed_vocab,labels

    def write_dict_csv(self,dict,name):
        import csv
        with open(name, 'wb') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in dict.items():
                writer.writerow([key,unicode(value).encode("utf-8")])

    def write_list_csv(self,list,name):
        import csv
        with open(name, 'wb') as csv_file:
            writer = csv.writer(csv_file)
            for item in list:
                writer.writerow(item)





AraSent=AraTweetReversed()
(body, rating, vocab)=AraSent.read_clean_reviews()


reversed_reviews,reversed_vocab,labels = AraSent.reverse_dataset(body,rating,vocab)
print labels
print "writing"
AraSent.write_list_csv(reversed_reviews,"reversed_reviews.csv")
AraSent.write_dict_csv(reversed_vocab,"reversed_vocab.csv")
AraSent.write_list_csv(labels,"labels.csv")

# AraSent.split_train_validation_test_no_obj(rating,0.2, 0.2,"unbalanced")
# AraSent.split_train_validation_test_no_obj(rating,0.2, 0.2,"balanced")
# print AraSent.split_train_validation_test(rating,0.2, 0.2,"unbalanced")

