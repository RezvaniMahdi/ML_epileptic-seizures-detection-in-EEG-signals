import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn import metrics, preprocessing
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from glob import glob
import random
import os
import collections
import math
from sklearn import tree
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
seed = 57
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
np.set_printoptions(suppress=True)

'''this method give us the element of each cluster'''
def ClusterIndicesNumpy(clustNum, labels_array): #numpy
    return np.where(labels_array == clustNum)[0]


def feature_selection(df,y, columns):
    feature_selected = []
    accuracy = []
    for i in range(15):
        '''calculate accuracy'''
        data = df[columns[i]]
        data = np.array(data)
        data = data.reshape(500, 1)    #Change this if your size input changed

        x_train, x_test, y_train, y_test = train_test_split(data,y,random_state=seed,test_size=0.2)
        clf = tree.DecisionTreeClassifier(random_state=seed)
        clf = clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)

        accuracy.append(accuracy_score(y_test, y_pred))

    feature_selected.append(df[columns[accuracy.index(max(accuracy))]])
    # print(accuracy)
    acc = 0
    old_acc = -1
    r = 0
    '''we continue untill if we add new feature our accuracy be less than before'''
    while acc >= old_acc:
        '''here we calculate corelation of new features into features selected'''
        merge_acc_corr = []
        j = 0
        for element in df:
            temp = []
            for h in feature_selected:
                cor = df[element].corr(h)
                cor = abs(cor)
                cor = 1 - cor
                temp.append(cor)
            corelation = min(temp)
            merge = (2 * corelation * accuracy[j]) / (corelation + accuracy[j])
            merge_acc_corr.append(merge)
            j += 1
        '''if we add this new feature to features selected our accuracy how change'''
        temp = []
        for ele in feature_selected:
            temp.append(ele)
        temp.append(df[columns[merge_acc_corr.index(max(merge_acc_corr))]])
        temp = np.array(temp).T
        x_trainn, x_tesst, y_trainn, y_tesst = train_test_split(temp,y,random_state=seed,test_size=0.2)
        clf = tree.DecisionTreeClassifier(random_state=seed)
        clf = clf.fit(x_trainn,y_trainn)
        y_predd = clf.predict(x_tesst)
        old_acc = acc
        acc = accuracy_score(y_tesst, y_predd)
        '''if new accuracy is less than before accuracy we dont join new feature to feature selected and done our work'''
        if acc < old_acc:
            break

        feature_selected.append(df[columns[merge_acc_corr.index(max(merge_acc_corr))]])


    '''we have to change place of row and columns becouse each row are a feature'''
    '''in other way the shape of array is (y,x) we change it to (x,y)'''
    feature_selected = np.array(feature_selected).T
    # print(feature_selected.shape)
    '''here we split our data and implement kmeans for clustring our train data'''
    x_train, x_test, y_train, y_test = train_test_split(feature_selected, y, random_state=seed, test_size=0.2)
    num_cluster = 5
    kmeans = KMeans(n_clusters=num_cluster, random_state=seed, n_init="auto").fit(x_train)
    '''we have 5 classifier becouse we have 5 cluster'''
    clf_cluster0 = RandomForestClassifier(max_depth=5, random_state=seed, bootstrap=True)
    clf_cluster1 = RandomForestClassifier(max_depth=5, random_state=seed, bootstrap=True)
    clf_cluster2 = RandomForestClassifier(max_depth=5, random_state=seed, bootstrap=True)
    clf_cluster3 = RandomForestClassifier(max_depth=5, random_state=seed, bootstrap=True)
    clf_cluster4 = RandomForestClassifier(max_depth=5, random_state=seed, bootstrap=True)
    classifiers = []
    classifiers.append(clf_cluster0)
    classifiers.append(clf_cluster1)
    classifiers.append(clf_cluster2)
    classifiers.append(clf_cluster3)
    classifiers.append(clf_cluster4)
    # print(clus.predict(x_test))

    '''here we make a classfier for each cluster'''
    for i in range(num_cluster):
        '''here we get the index of element that belongs to i cluster'''
        cluster_i = ClusterIndicesNumpy(i, kmeans.labels_)
        # print(cluster_i.shape)
        xtrain = []
        ytrain = []
        for element in cluster_i:
            xtrain.append(x_train[element])
            ytrain.append(y_train[element])
        xtrain = np.array(xtrain)
        ytrain = np.array(ytrain)
        '''fit our classfiers for each cluster'''
        classifiers[i].fit(xtrain,ytrain)

    '''here we analyse our tests'''
    kmeans_labels_x_test = kmeans.predict(x_test)
    for i in range(num_cluster):
        xtest = []
        ytest = []
        cluster_id = ClusterIndicesNumpy(i, kmeans_labels_x_test)
        # print(cluster_id.shape)
        for element in cluster_id:
            xtest.append(x_test[element])
            ytest.append(y_test[element])
        xtest = np.array(xtest)
        ytest = np.array(ytest)
        predic_y = classifiers[i].predict(xtest)
        print(accuracy_score(ytest, predic_y,))
        print(precision_score(ytest,predic_y))
        print(recall_score(ytest, predic_y))
        print("-------------")





'''we use pandas dataframe because its make easy to calculate corelation'''
def state_2_class_1():
    x = pickle.load(open('signal_features_extracted.pkl', 'rb'))
    y = np.concatenate((np.zeros((400, 1)), np.ones((100, 1))))
    columns = ['mean', 'median', 'variance', 'std', 'ptp', 'mobility', 'complexity', 'skewness', 'kurtosis', 'dfa',
               'pfd', 'hurst', 'approximate_entropy', 'shanon', 'permutation_entropy']
    df = pd.DataFrame(x, columns=columns)
    feature_selection(df, y, columns)


def state_2_class_2():
    x = pickle.load(open('signal_features_extracted.pkl', 'rb'))
    y = np.concatenate((np.zeros((100, 1)), np.ones((100, 1))))
    x = np.concatenate((x[:100], x[400:]), axis=0)   # set D (F) and set E (S)
    columns = ['mean', 'median', 'variance', 'std', 'ptp', 'mobility', 'complexity', 'skewness', 'kurtosis', 'dfa',
               'pfd', 'hurst', 'approximate_entropy', 'shanon', 'permutation_entropy']
    df = pd.DataFrame(x, columns=columns)
    feature_selection(df, y, columns)

def state_2_class_3():
    x = pickle.load(open('signal_features_extracted.pkl', 'rb'))
    y = np.concatenate((np.zeros((100, 1)), np.ones((100, 1))))
    x = np.concatenate((x[200:300], x[400:]), axis=0)   # set C (N) and set E (S)
    columns = ['mean', 'median', 'variance', 'std', 'ptp', 'mobility', 'complexity', 'skewness', 'kurtosis', 'dfa',
               'pfd', 'hurst', 'approximate_entropy', 'shanon', 'permutation_entropy']
    df = pd.DataFrame(x, columns=columns)
    feature_selection(df, y, columns)


def state_3_class():
    x = pickle.load(open('signal_features_extracted.pkl', 'rb'))
    y = []
    for i in range(200):
        y.append(0)
    for i in range(200):
        y.append(1)
    for i in range(100):
        y.append(2)
    y = np.array(y)
    y = y.reshape(500,1)
    columns = ['mean', 'median', 'variance', 'std', 'ptp', 'mobility', 'complexity', 'skewness', 'kurtosis', 'dfa',
               'pfd', 'hurst', 'approximate_entropy', 'shanon', 'permutation_entropy']
    df = pd.DataFrame(x, columns=columns)
    feature_selection(df, y, columns)
