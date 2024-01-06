"""
Author: Shubham Raheja (shubhamraheja1999@gmail.com)
Code Description: A python code to tune the hyperparameters of SVM on the AR+FT dataset.
"""
import os
import numpy as np
import pickle
import joblib
from scipy import fft as fftpack
from utils import load_pkl

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, accuracy_score, classification_report)


'''
_______________________________________________________________________________

Rule used for renaming class labels: (Class Label - Numeric Code)
_______________________________________________________________________________

    AR+FT series        -     0
    Random+FT series     -     1
_______________________________________________________________________________

Variable description:
_______________________________________________________________________________

    ar_dataset      -   AR dataset.
    random_dataset  -   Random dataset.     
    X               -   Data attributes.
    y               -   Corresponding labels for X.
    X_train         -   Data attributes for training (80% of the dataset).
    y_train         -   Corresponding labels for X_train.
    X_test          -   Data attributes for testing (20% of the dataset).
    y_test          -   Corresponding labels for X_test.
    X_train_norm    -   Normalizised training data attributes (X_train).
    X_test_norm     -   Normalized testing data attributes (X_test).
_______________________________________________________________________________

ML hyperparameter description:
_______________________________________________________________________________

    c   -   Regularization parameter. The strength of the regularization is 
    inversely proportional to C. Must be strictly positive. The penalty is a 
    squared l2 penalty.
    Source: Scikit Learn 
    (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
_______________________________________________________________________________

Performance metric used:
_______________________________________________________________________________

    Macro F1-score (F1SCORE/ f1) -
    The F1 score can be interpreted as a harmonic mean of the precision and 
    recall, where an F1 score reaches its best value at 1 and worst score at 
    0. The relative contribution of precision and recall to the F1 score are 
    equal; 'macro' calculates metrics for each label, and find stheir 
    unweighted mean. This does not take label imbalance into account.
    Source: Scikit Learn 
    (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
_______________________________________________________________________________

''' 
# Import the AR and Random Datasets 
length = 75

ar_dataset = load_pkl(f'path/Length_{length}/ar_dataset.pkl')
random_dataset = load_pkl(f'path/Length_{length}/rand_dataset.pkl')

# FFT of AR and Random datasets
freq_ar_dataset = np.abs(fftpack.fftfreq(length, 1)) # Discrete Fourier Transform sample frequencies
fft_ar_dataset = fftpack.fft(ar_dataset)
ampli_ar_dataset = 2/length * np.abs(fft_ar_dataset) 
freq_random_dataset = np.abs(fftpack.fftfreq(length, 1))
fft_random_dataset = fftpack.fft(random_dataset)
ampli_random_dataset = 2/length * np.abs(fft_random_dataset)

# Creating labels for AR and Random datasets: Random - 0, AR - 1
y_ar = np.ones((1250,1))
y_ran = np.zeros((1250,1))

# Reading data and labels from the dataset
X = np.concatenate((ampli_random_dataset, ampli_ar_dataset))
y = np.concatenate((y_ran,y_ar))

# Splitting the dataset for training, validation and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# Algorithm - SVM
BESTF1 = 0
FOLD_NO = 5
KF = KFold(n_splits= FOLD_NO, random_state=42, shuffle=True)  
KF.get_n_splits(X_train) 
print(KF) 
for c in np.arange(0.1, 5, 0.1):
    FSCORE_TEMP=[]
    
    for TRAIN_INDEX, VAL_INDEX in KF.split(X_train):
        
        X_TRAIN, X_VAL = X_train[TRAIN_INDEX], X_train[VAL_INDEX]
        Y_TRAIN, Y_VAL = y_train[TRAIN_INDEX], y_train[VAL_INDEX]
    
        clf = SVC(C = c, kernel='rbf', decision_function_shape='ovr', random_state=42)
        clf.fit(X_TRAIN, Y_TRAIN.ravel())
        Y_PRED = clf.predict(X_VAL)
        f1 = f1_score(Y_VAL, Y_PRED, average='macro')
        FSCORE_TEMP.append(f1)
        print('F1 Score', f1)
    print("Mean F1-Score for C = ", c," is  = ",  np.mean(FSCORE_TEMP)  )
    if(np.mean(FSCORE_TEMP) > BESTF1):
        BESTF1 = np.mean(FSCORE_TEMP)
        BESTC = c
        
print("BEST F1SCORE", BESTF1)
print("BEST C = ", BESTC)

# Build the SVM model with best hyperparameters
clf = SVC(C = BESTC, kernel='rbf', decision_function_shape='ovr', random_state=42)
clf.fit(X_train, y_train.ravel())

# Make predictions with trained model on train data
y_pred_traindata = clf.predict(X_train)
ACC = accuracy_score(y_train, y_pred_traindata)*100
F1SCORE_train = f1_score(y_train, y_pred_traindata, average="macro")
print('TRAIN: ACCURACY = ', ACC , " F1 SCORE = ", F1SCORE_train)
print(classification_report(y_train, y_pred_traindata))

y_pred_testdata = clf.predict(X_test)
ACC = accuracy_score(y_test, y_pred_testdata)*100
F1SCORE_test = f1_score(y_test, y_pred_testdata, average="macro")
print('TEST: ACCURACY = ', ACC , " F1 SCORE = ", F1SCORE_test)
print(classification_report(y_test, y_pred_testdata))

print("Saving Hyperparameter Tuning Results")

PATH = os.path.dirname(__file__)
RESULT_PATH = PATH + f'/SVM-TUNING/AR_FT_Hyperparameters/Length_{length}'
RESULT_PATH_FINAL = PATH + f'/TESTING-RESULTS/On_Training_set/AR_FT_Results/Length_{length}'
RESULT_PATH_SVM = PATH + '/SVM-logs/AR_FT_Checkpoints/'

try:
    os.makedirs(RESULT_PATH)
except OSError:
    print ("Creation of the result directory %s not required" % RESULT_PATH)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH)

try:
    os.makedirs(RESULT_PATH_FINAL)
except OSError:
    print ("Creation of the result directory %s not required" % RESULT_PATH_FINAL)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH_FINAL)

try:
    os.makedirs(RESULT_PATH_SVM)
except OSError:
    print ("Creation of the result directory %s not required" % RESULT_PATH_SVM)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH_SVM)

# Saving hyperparameters
np.save(RESULT_PATH+"/h_C.npy", np.array([BESTC]) ) 

# Saving results
np.save(RESULT_PATH_FINAL+ "/Train_F1SCORE.npy", (F1SCORE_train))
np.save(RESULT_PATH_FINAL+ "/Test_F1SCORE.npy", (F1SCORE_test))

# Saving model
joblib.dump(clf, os.path.join(RESULT_PATH_SVM, f"svm_{length}.pkl")) 