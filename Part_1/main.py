import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
from sklearn import metrics, preprocessing
import pickle
from scipy.signal import butter, lfilter
from scipy.stats import stats
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from glob import glob
import random
import os
import pyeeg
import collections
import math

seed = 57
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
np.set_printoptions(suppress=True)


def normalize(input):
    result = preprocessing.normalize(input)
    return result


'''implement shanon entropy'''
def shanon_entropy(s):
    probabilities = [n_x/len(s) for x,n_x in collections.Counter(s).items()]
    e_x = [-p_x*math.log(p_x,2) for p_x in probabilities]
    return sum(e_x)

'''we extracted 15 features (Time domain and statistical and entropy and non-linear) from our EEG signals'''
'''the name of each feature is infront of his method'''
def feature_extraction(signals):
    features = []
    '''in each itaration we calculate 15 features for a signal (we have 500 segment so our itaration is 500) '''
    '''we use round() function to round our numbers'''
    for i in range(signals.shape[0]):
        data = []
        data.append(round(np.mean(signals[i]), 4))                         # mean
        data.append(round(np.median(signals[i]), 4))                       # median
        data.append( round(np.var(signals[i]), 4))                         # variance
        data.append(round(np.std(signals[i]), 4))                          # std
        data.append(round(np.ptp(signals[i]), 4))                          # ptp
        mobility_complexity = pyeeg.hjorth(signals[i])                     # calculate mobility and complexity
        data.append(round(mobility_complexity[0], 4))                      # mobility
        data.append(round(mobility_complexity[1], 4))                      # complexity
        data.append(round(stats.skew(signals[i],axis=0,bias=True), 4))     # skewness
        data.append(round(stats.kurtosis(signals[i], axis=0,bias=True), 4))# kurtosis
        data.append(round(pyeeg.dfa(signals[i]), 4))                       # dfa
        data.append(round(pyeeg.pfd(signals[i]), 4))                       # pfd
        data.append(round(pyeeg.hurst(signals[i]), 4))                     # hurst
        data.append(round(pyeeg.ap_entropy(signals[i],2, 0.2), 4))         # approximate_entropy
        data.append(round(shanon_entropy(signals[i]), 4))                  # shanon entropy
        data.append(round(pyeeg.permutation_entropy(signals[i], 3, 1), 4)) # permutation_entropy
        features.append(data)
    return features

def extraction():
    x = pickle.load(open('x.pkl', 'rb'))
    y = pickle.load(open('y.pkl', 'rb'))
    x_normal = np.concatenate((x[:300], x[400:]), axis=0)
    x_seizure = x[300:400]
    # print(x_normal.shape)
    # print(x_seizure.shape)
    sampling_freq = 173.6  # based on info from website
    b, a = butter(3, [0.5,40], btype='bandpass',fs=sampling_freq)
    x_normal_filtered = np.array([lfilter(b,a,x_normal[ind,:]) for ind in range(x_normal.shape[0])])
    x_seizure_filtered = np.array([lfilter(b,a,x_seizure[ind,:]) for ind in range(x_seizure.shape[0])])
    #print(x_normal.shape)
    #print(x_seizure.shape)
    x_normal = x_normal_filtered
    x_seizure = x_seizure_filtered
    x = np.concatenate((x_normal,x_seizure))    # last 100 signals are seizures
    y = np.concatenate((np.zeros((400, 1)), np.ones((100, 1))))

    '''from here new code is begin that call features_extraction method and extract features from our signals'''
    '''and save result in pickle file and text file '''

    extraction = feature_extraction(x)
    extraction = np.array(extraction)
    pickle.dump(extraction, open('signal_features_extracted.pkl' , 'wb'))
    # print(extraction.shape)
    file = open("features_extracted.txt","a")
    for i in extraction:
        file.write(str(i))
        file.write("\n")



'''in this function we use kfold with k == 5 to split our dataset to train set and test set for classifier'''
def kfold_split():
    y = np.concatenate((np.zeros((400, 1)), np.ones((100, 1))))
    x = pickle.load(open('signal_features_extracted.pkl', 'rb'))
    # x = normalize(x)
    kf = KFold(5, random_state=seed, shuffle=True)
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    '''here k-fold give us the index of each train and test element so we use them to take our data from orginal array'''
    for i, (train_index, test_index) in enumerate(kf.split(x)):
        x_train.append(x[train_index])
        y_train.append(y[train_index])
        x_test.append(x[test_index])
        y_test.append(y[test_index])

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # print(x_train.shape)
    # print(x_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)

    return x_train,x_test,y_train,y_test



def svm_classifier():
    x_train, x_test, y_train, y_test = kfold_split()
    accuracy = []
    precision = []
    recall = []
    for i in range(5):
        clf = SVC(kernel='rbf',random_state=seed, probability=True)
        clf.fit(x_train[i], y_train[i])
        y_pred = clf.predict(x_test[i])
        accuracy.append(accuracy_score(y_test[i], y_pred))
        precision.append(precision_score(y_test[i], y_pred))
        recall.append(recall_score(y_test[i], y_pred))
    accuracy = np.array(accuracy)
    precision = np.array(precision)
    recall = np.array(recall)
    print(np.mean(accuracy))
    print(np.var(accuracy))
    print(np.mean(precision))
    print(np.var(precision))
    print(np.mean(recall))
    print(np.var(recall))




def random_forest_classifier():
    x_train, x_test, y_train, y_test = kfold_split()
    accuracy = []
    precision = []
    recall = []
    for i in range(5):
        clf = RandomForestClassifier(n_estimators=100,max_depth=3, random_state=seed, bootstrap=True)
        clf.fit(x_train[i], y_train[i])
        y_pred = clf.predict(x_test[i])
        # print(metrics.confusion_matrix(y_test[i], y_pred))  # calculate cofusion matrix
        accuracy.append(accuracy_score(y_test[i], y_pred))
        precision.append(precision_score(y_test[i], y_pred))
        recall.append(recall_score(y_test[i], y_pred))
    accuracy = np.array(accuracy)
    precision = np.array(precision)
    recall = np.array(recall)
    print(np.mean(accuracy))
    print(np.var(accuracy))
    print(np.mean(precision))
    print(np.var(precision))
    print(np.mean(recall))
    print(np.var(recall))

def knn_classifier():
    x_train, x_test, y_train, y_test = kfold_split()
    accuracy = []
    precision = []
    recall = []
    for i in range(5):
        clf = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
        clf.fit(x_train[i], y_train[i])
        y_pred = clf.predict(x_test[i])
        accuracy.append(accuracy_score(y_test[i], y_pred))
        precision.append(precision_score(y_test[i], y_pred))
        recall.append(recall_score(y_test[i], y_pred))
    accuracy = np.array(accuracy)
    precision = np.array(precision)
    recall = np.array(recall)
    print(np.mean(accuracy))
    print(np.var(accuracy))
    print(np.mean(precision))
    print(np.var(precision))
    print(np.mean(recall))
    print(np.var(recall))

def roc_curve_random_forest():
    x_train, x_test, y_train, y_test = kfold_split()
    for i in range(5):
        clf = RandomForestClassifier(max_depth=3, random_state=seed, bootstrap=True)
        clf.fit(x_train[i], y_train[i])
        y_pred = clf.predict_proba(x_test[i])[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(y_test[i], y_pred)
        roc_auc = metrics.auc(fpr, tpr)
        print('roc_auc_score for Random Forest: ', metrics.roc_auc_score(y_test[i], y_pred))
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="Random forest")
        display.plot()
        plt.show()

def draw_plot():
    y = [0.918, 0.903, 0.909,0.899]
    x = np.array([0, 1, 2, 3])
    my_xticks = ['3', '5', '7', '13']
    plt.xlabel("K-Value")
    plt.xticks(x, my_xticks)
    plt.plot(x, y)
    plt.title("F-Score")
    plt.show()

# svm_classifier()
# random_forest_classifier()
# knn_classifier()