

import numpy as np
from symbol import return_stmt
import matplotlib.pyplot as plt
import codecs
import re

def show_most_informative_features(vectorizer, clf, file_name, n=20):
    out_file = open(file_name + "_" + str(n) + ".txt", 'w', buffering=100)
    c_f = sorted(zip(clf.coef_[0], vectorizer.get_feature_names()))
    top = zip(c_f[:n], c_f[:-(n + 1):-1])
    for (c1, f1), (c2, f2) in top:
        line = ("%-15s\n" % (f1))
        line =re.sub(("^\s+"), "", line)
        line =re.sub(("\s+$"), "", line)
        out_file.write(line+'\n')
    for (c1, f1), (c2, f2) in top:
        line = ("%-15s\n" % ( f2))
        line =re.sub(("^\s+"), "", line)
        line =re.sub(("\s+$"), "", line)
        out_file.write(line+'\n')
def MySelectPercentile(vectorizer,feat_ext, precent, X_train, y_train, X_test):
    name=feat_ext['name']
    feat_ext=feat_ext['feat_ext']
    
    # fit the classifier
    feat_ext.fit(X_train, y_train) 
    # total number of features
    N = len(feat_ext.coef_[0]) 
    # sort the coeffiecints in descending order
    c_f = sorted(zip(feat_ext.coef_[0], range(0, N)), reverse=True) 
    
    # get indicies of the features in the sorted order    
    Indeces = [x[1] for x in c_f] 
    # precentage of features to be selected
    N_selected = int (round((precent / 100.0) * N))
    
#     show_most_informative_features(vectorizer, feat_ext, name, N_selected)
    
    print ("N_selected : ", N_selected)
    
    Indeces_Selected_positive = Indeces[:N_selected / 2]
    Indeces_Selected_negative = Indeces[N_selected / (-2):]
    
    Indeces_Selected = np.concatenate((Indeces_Selected_positive, Indeces_Selected_negative))
    X_train_modified = X_train[:, Indeces_Selected]
    X_test_modified = X_test[:, Indeces_Selected]
    return X_train_modified, X_test_modified

def stackedbarplot(x_data, y_data_list, colors, y_data_names="", x_label="", y_label="", title=""):
    _, ax = plt.subplots()
    # Draw bars, one category at a time
    for i in range(0, len(y_data_list)):
        print i
        if i == 0:
            ax.bar(x_data, y_data_list[i], color = colors[i], align = 'center')
        else:
            # For each category after the first, the bottom of the
            # bar will be the top of the last category
            ax.bar(x_data, y_data_list[i], color = colors[i], bottom = y_data_list[i - 1], align = 'center')
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc = 'upper right')


    plt.show()


def Evaluate_Result(pred, y_test):
    # Weighted average of accuracy and f1
    (acc, tacc, support, f1) = (list(), list(), list(), list())

    total = []
    true = []

    classes_accuracy = []

    for l in np.unique(y_test):
        support.append(np.sum(y_test == l) / float(y_test.size))

        tp = float(np.sum(pred[y_test == l] == l))
        fp = float(np.sum(pred[y_test != l] == l))
        fn = float(np.sum(pred[y_test == l] != l))
        # tn = float(np.sum(pred[y_test != l] != l))

        total.append(int(fn))
        true.append(int(tp))

        classes_accuracy.append(tp)
        # classes_accuracy.append(tp/(tp + fn))

        #print("tp:", tp, " fp:", fp, " fn:", fn,"class:",l,"precision:",tp/(tp+fp),"recall:",tp/(tp+fn))
        if tp > 0:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
        else:
            (prec, rec) = (0, 1)
        
        f1.append(2 * prec * rec / (prec + rec))
        acc.append(tp / float(np.sum(y_test == l)))
        tacc.append(tp)
                        
    # compute total accuracy
    tacc = np.sum(tacc) / y_test.size
    # weighted accuracy
    acc = np.average(acc, weights=support)
    # weighted F1 measure
    f1 = np.average(f1, weights=support)

    print("f1 = %0.3f" % f1)
    print("wacc = %0.3f" % acc)
    print("tacc = %0.3f" % tacc)
    # stackedbarplot(["negative", "neutral", "positive"], [true, total], ["red", "blue"])
    return (acc, tacc, support, f1, classes_accuracy)

def Train_And_Predict(X_train, y_train, X_test, classifier, classifier_name):
    
####################################Training########################################                        
    print("Training: ", classifier_name)
    classifier.fit(X_train, y_train)
####################################Testing#########################################
    print("Testing")              
    # for knn predict patches of patterns to save memory
    if(classifier_name == 'KNN'):         
        n = X_test.shape[0]
        patch_size = 100
        div = n / patch_size
        pred = np.array([])
        for  i in range (0, div):
            X_test_patch = X_test[(i * patch_size):(((i + 1) * patch_size)), :]
            pred_patch = classifier.predict(X_test_patch)
            pred = np.concatenate((pred, pred_patch))
        if (div * patch_size < n):
            X_test_patch = X_test[div * patch_size:n, :]
            pred_patch = classifier.predict(X_test_patch)
            pred = np.concatenate((pred, pred_patch))
    else:
        pred = classifier.predict(X_test)
    return pred

def plot(accuracies, precentages, legend_names, feat_extract):
    fig = plt.figure()
    subplot = fig.add_subplot(111)
    color_ = ["g", "b", "r", "c", "m", "y", "b"]
    for i in range(0, accuracies.shape[0]):
        subplot.plot(range(1,len(precentages)+1), accuracies[i,:], color=color_[i],
                marker="D", label=legend_names[i])
#     subplot.plot(range(1,len(precentages)+1), accuracies, color=color_[0],
#                 marker="D", label=legend_names)
    plt.xlabel("Precentage of Features")
    
    plt.xticks(range(1,len(precentages)+1))
    plt.gca().set_xticklabels(precentages)

    
    plt.ylabel("F-Measure")
    plt.title("F1 measure for < " + feat_extract + " > vs Precentage of Features")

    # #set legend box
    box = subplot.get_position()
    subplot.set_position([box.x0, box.y0 + box.height * 0.3,
                     box.width, box.height * 0.7])
    subplot.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=3)
    
    plt.draw()


def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


    
def ReadLexicon():
    pos = codecs.open('../data/lexicon/Pos.txt', 'r', 'utf-8').readlines()
    neg = codecs.open('../data/lexicon/Neg.txt', 'r', 'utf-8').readlines()
    for i in range(0,len(pos)):
        pos[i]=re.sub('\s$','',pos[i])
    for i in range(0,len(neg)):
        neg[i]=re.sub('\s$','',neg[i])        
    lexicon=pos+neg
    return lexicon
def ReadLexicon1():
    pos = codecs.open('../data/sam_lex/Pos.txt', 'r', 'utf-8').readlines()
    neg = codecs.open('../data/sam_lex/Neg.txt', 'r', 'utf-8').readlines()
    for i in range(0,len(pos)):
        pos[i]=re.sub('\s$','',pos[i])
    for i in range(0,len(neg)):
        neg[i]=re.sub('\s$','',neg[i])        
    lexicon=pos+neg
    return lexicon

def stop_word_perc(d_train_ALL,y_train_ALL,stopwords_list):
    neutral_indices = [i for i, x in enumerate(y_train_ALL) if x == 'NEUTRAL']
    postive_indices = [i for i, x in enumerate(y_train_ALL) if x == "POS"]
    negtive_indices = [i for i, x in enumerate(y_train_ALL) if x == "NEG"]

    neutral_stop_words = 0
    postive_stop_words = 0
    negtive_stop_words = 0

    neutral_total_words = 0
    postive_total_words = 0
    negtive_total_words = 0

    for index in neutral_indices:

        for word in d_train_ALL[index].split(' '):
            neutral_total_words = neutral_total_words + 1
            matching = [s for s in stopwords_list if s == word]
            # if(len(matching)> 0):
            #     print matching[0]
            #     print d_train_ALL[index]
            neutral_stop_words = neutral_stop_words + len(matching)

    print "neutral stop words " + str(neutral_stop_words) + "  neutral total words " + str(
        neutral_total_words) + "   " + str(neutral_stop_words * 1.0 / neutral_total_words)

    for index in postive_indices:

        for word in d_train_ALL[index].split(' '):
            postive_total_words = postive_total_words + 1
            matching = [s for s in stopwords_list if s == word]
            # if (len(matching) > 0):
            #     print matching[0]
            #     print d_train_ALL[index]
            postive_stop_words = postive_stop_words + len(matching)

    print "postive stop words " + str(postive_stop_words) + "  postive total words " + str(
        postive_total_words) + "   " + str(postive_stop_words * 1.0 / postive_total_words)

    for index in negtive_indices:

        for word in d_train_ALL[index].split(' '):
            negtive_total_words = negtive_total_words + 1
            matching = [s for s in stopwords_list if s == word]
            # if (len(matching) > 0):
            #     print matching[0]
            #     print d_train_ALL[index]
            negtive_stop_words = negtive_stop_words + len(matching)

    print "negtive stop words " + str(negtive_stop_words) + "  negtive total words " + str(
        negtive_total_words) + "   " + str(negtive_stop_words * 1.0 / negtive_total_words)



def groupedbarplot(labels,bars,types,colors,title,suptitle,total,barWidth = 0.5):

    """
    :param labels: classifiers names ex: ['LRegn', 'PAgg', 'SVM', 'Percep', 'bnb', 'mnb', 'sgd', 'KNN']
    :param bars: true positive for each class ex: [(282.0, 241.0, 251.0, 224.0, 332.0, 265.0, 254.0, 43.0), (23.0, 34.0, 31.0, 44.0, 1.0, 32.0, 37.0, 3.0), (76.0, 83.0, 78.0, 79.0, 6.0, 72.0, 79.0, 146.0)]
    :param types: classification types ex: ['Negative', 'Neutral', 'Positive']
    :param colors: colors of classification types ex: ['red', 'green', 'blue']
    :param title: type of dataset ex: 4-unbalanced
    :param suptitle: type of feature extraction ex: count_ng1
    :param total: total number in each class ex: [336, 166, 159]

    """

    # libraries
    import numpy as np
    import matplotlib.pyplot as plt

    # Set position of bar on X axis
    x_postions =  []
    r = np.arange(len(bars[0]))*(len(types) * barWidth + barWidth)
    for i in range(len(bars)):
        x_postions.append([x + barWidth*i for x in r])


    ax = plt.subplot()
    # Plot bars on Y axis
    for i in range(len(bars)):
        if(i < 3):
            ax.bar(x_postions[i], bars[i], color=colors[i], width=barWidth, edgecolor='white', label=types[i]+": "+str(total[i]))
        else:
            ax.bar(x_postions[i], bars[i], color=colors[i%3], width=barWidth, edgecolor='white')


    # Add labels on the middle of the group bars

    ax.set_xlabel('group', fontweight='bold')
    ax.set_xticks(x_postions[len(x_postions)/2])
    ax.set_xticklabels(labels)

    ax.set_title(title)
    plt.suptitle(suptitle)

    # Create legend & Show graphic
    plt.legend()

    # Add numbers on the top of bars
    rects = ax.patches
    labels = list(np.array((bars)).ravel())
    labels = [int(x) for x in labels]

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 3, label,
                ha='center', va='bottom')

    plt.show()