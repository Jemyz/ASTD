# # -*- coding: utf-8 -*-
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

TYPES = ["Negative","Neutral","Positive"]
COLORS = ["red","green","blue"]

gr = AraTweet()
scores = list()



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

    from sklearn.utils import shuffle


    d_train_ALL, y_train_ALL = shuffle(d_train_ALL, y_train_ALL)
    d_test_ALL, y_test_ALL = shuffle(d_test_ALL, y_test_ALL)


    neutral_indices = [i for i, x in enumerate(y_test_ALL) if x == 'NEUTRAL' ]
    postive_indices = [i for i, x in enumerate(y_test_ALL) if x == "POS" ]
    negtive_indices = [i for i, x in enumerate(y_test_ALL) if x == "NEG" ]

    TOTALS = [len(negtive_indices),len(neutral_indices),len(postive_indices)]

    stop_word_perc(d_train_ALL,y_train_ALL,stopwords_list)

    ##### for delta-tfidf
    indices_train = [i for i, x in enumerate(y_train_ALL) if x == "POS" or x == "NEG" or x == 'NEUTRAL' ]
    indices_test = [i for i, x in enumerate(y_test_ALL) if x == "POS" or x == "NEG" or x == 'NEUTRAL' ]

    # indices_neg = [i for i, x in enumerate(y_train) if x == "NEG"]
    x_train_PN = np.array(d_train_ALL[indices_train])
    y_train_PN = np.array(y_train_ALL[indices_train])
    x_test_PN = np.array(d_test_ALL[indices_test])
    y_test_PN = np.array(y_test_ALL[indices_test])



    y_train_PN[y_train_PN == 'POS'] = int(1)
    y_train_PN[y_train_PN == 'NEG'] = int(-1)
    y_train_PN[y_train_PN == 'NEUTRAL'] = int(0)
    y_train_PN = map(int, y_train_PN)

    y_test_PN[y_test_PN == 'POS'] = int(1)
    y_test_PN[y_test_PN == 'NEG'] = int(-1)
    y_test_PN[y_test_PN == 'NEUTRAL'] = int(0)
    y_test_PN = np.array(map(int, y_test_PN))
    ################################################

    ####################################################################################
    for feat_generator in Features_Generators:
        ####################################Features Generation#############################
        print("Features Generation:", feat_generator['name'])

        x_train = d_train_ALL
        y_train = y_train_PN
        x_test = d_test_ALL
        y_test = y_test_PN


        if (feat_generator['name'].startswith('delta_tfidf')):

            X_train = feat_generator['feat_generator'].fit_transform(x_train, y_train)
            X_test = feat_generator['feat_generator'].transform(x_test.astype('U'))
        else:

            X_train = feat_generator['feat_generator'].fit_transform(x_train)
            X_test = feat_generator['feat_generator'].transform(x_test)

        ####################################################################################

        clfs_names = []
        classes_accuracies = []
        for clf in classifiers:
                if clf["name"] == 'mnb' and (feat_generator['name'].startswith('hash_ng') or feat_generator['name'].startswith('delta_tfidf')):
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

                    (acc, tacc, support, f1, classes_accuracy) = Evaluate_Result(pred, y_test)

                    clfs_names.append(clf['name'])
                    classes_accuracies.append(classes_accuracy)

                    score = dict(data=data['name'],
                                     feat_generator=feat_generator['name'],
                                     clf=clf['name'],
                                      # feat_ext=feat_ext['name'],
                                     f1=f1,
                                     acc=acc,
                                     tacc=tacc)

                    scores.append(score)
        groupedbarplot(clfs_names,zip(*classes_accuracies),TYPES,COLORS,data['name'],feat_generator['name'],TOTALS)
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


import pandas as pd
df = pd.DataFrame(scores)  # transpose to look just like the sheet above
df.to_csv('results_union.csv')
df.to_excel('file.xls')

