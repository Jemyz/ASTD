# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 19:05:12 2013

@author1: Mohamed Aly <mohamed@mohamedaly.info>
@author2: Mahmoud Nabil <mah.nabil@yahoo.com>

"""

from Definations import *
from Utilities import *
from sklearn.feature_selection.univariate_selection import SelectPercentile
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.manifold import isomap
import numpy as np
from scipy.sparse import hstack

gr = AraTweet()
scores = list()



# from sklearn_deltatfidf import DeltaTfidfVectorizer
#
# v = DeltaTfidfVectorizer()
# data = [u'word1 word2', u'word2', u'word2 word3', u'word4']
# labels = [u'1', u'-1', u'-1', u'1']
# v.fit_transform(data, labels)
# exit()

for data in datas:
    ###################################load the data####################################
    print(60 * "-")
    print("Loading data:", data['name'])

    if (LoadValidation):
        (d_train_ALL, y_train_ALL, d_test_ALL, y_test_ALL, d_valid_ALL, y_valid_ALL) = gr.get_train_test_validation(**data['params'])
        if (Evaluate_On_TestSet):
            d_train_ALL = np.concatenate((d_train_ALL, d_valid_ALL))
            y_train_ALL = np.concatenate((y_train_ALL, y_valid_ALL))
        else:
            d_test_ALL = d_valid_ALL
            y_test_ALL = y_valid_ALL
    else:
        (d_train_ALL, y_train_ALL, d_test_ALL, y_test_ALL) = gr.get_train_test(**data['params'])

    ##### for delta-tfidf
    indices_train = [i for i, x in enumerate(y_train_ALL) if x == "POS" or x == "NEG"]
    indices_test = [i for i, x in enumerate(y_test_ALL) if x == "POS" or x == "NEG"]

    # indices_neg = [i for i, x in enumerate(y_train) if x == "NEG"]
    x_train_PN = (d_train_ALL[indices_train])
    y_train_PN = (y_train_ALL[indices_train])
    x_test_PN = (d_test_ALL[indices_test])
    y_test_PN = (y_test_ALL[indices_test])

    y_train_PN[y_train_PN == 'POS'] = int(1)
    y_train_PN[y_train_PN == 'NEG'] = int(-1)
    y_train_PN = map(int, y_train_PN)

    y_test_PN[y_test_PN == 'POS'] = int(1)
    y_test_PN[y_test_PN == 'NEG'] = int(-1)
    y_test_PN = np.array(map(int, y_test_PN))
    ################################################

    ####################################################################################

    for feat_generator in Features_Generators:
        ####################################Features Generation#############################
        print("Features Generation:", feat_generator['name'])
        if (feat_generator['name'].startswith('delta-tfidf')):
            x_train = x_train_PN
            y_train = y_train_PN
            x_test = x_test_PN
            y_test = y_test_PN
            X_train = feat_generator['feat_generator'].fit_transform(x_train, y_train)
            X_test = feat_generator['feat_generator'].transform(x_test.values.astype('U'))
        else:
            x_train = d_train_ALL
            y_train = y_train_ALL
            x_test = d_test_ALL
            y_test = y_test_ALL
            X_train = feat_generator['feat_generator'].fit_transform(x_train)
            X_test = feat_generator['feat_generator'].transform(x_test)

        ####################################################################################

        for clf in classifiers:
                if clf["name"] == 'mnb' and (feat_generator['name'].startswith('hash_ng') or feat_generator['name'].startswith('delta-tfidf')):
                    continue
                if clf['parameter_tunning']:
                    # region parameter tunning
                    print("tuning: ", clf["name"])
                    clf['tune_clf'].fit(X_train, y_train)
                    print (data['name'])
                    print (feat_generator['name'])
                    print (clf['tune_clf'].best_estimator_)
                    # endregion
                else:
                    ####################################Training And Predict################################
                    pred = Train_And_Predict(X_train, y_train, X_test, clf['clf'], clf["name"])

                    (acc, tacc, support, f1) = Evaluate_Result(pred, y_test)

                    score = dict(data=data['name'],
                                     feat_generator=feat_generator['name'],
                                     clf=clf['name'],
                                     # feat_ext=feat_ext['name'],
                                     f1=f1,
                                     acc=acc,
                                     tacc=tacc)

                    scores.append(score)
####################################Testing##############################################
value_unbalanced = 0.0
value_balanced = 0.0

print(60 * "=")
for s in scores:
    print
    for k, v in s.iteritems():
        if(k == "acc"):
            if(s["data"] == "4-unbalanced"):
                if(v > value_unbalanced):
                    value_unbalanced = v
                    temp_unbalanced = dict(s)
            else:
                if (v > value_balanced):
                    value_balanced = v
                    temp_balanced = dict(s)
        print(k, v)


print
print
print
print


for k, v in temp_unbalanced.iteritems():
    print(k, v)

print

for k, v in temp_balanced.iteritems():
    print(k, v)
