"""
Author: Shubham Raheja (shubhamraheja1999@gmail.com)
Code Description: A python code to tune the hyperparameters of RFC on the AR+FT+CFEX dataset.
"""
import os
import numpy as np
import pickle
import joblib
from scipy import fft as fftpack
from utils import load_pkl, scale_dataset, extractCFEX

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, accuracy_score, classification_report)
from Codes import k_cross_validation

'''
_______________________________________________________________________________

Rule used for naming class labels: (Class Label - Numeric Code)
_______________________________________________________________________________

    AR+FT+CFEX series         -     0
    Random+FT+CFEX series     -     1
_______________________________________________________________________________

Variable description:
_______________________________________________________________________________

    ar_dataset      -   Complete AR-Random dataset.
    random_dataset  -   Random dataset.    
    X               -   Data attributes.
    y               -   Corresponding labels for X.
    X_train         -   Data attributes for training (80% of the dataset).
    y_train         -   Corresponding labels for X_train.
    X_test          -   Data attributes for testing (20% of the dataset).
    y_test          -   Corresponding labels for X_test.
    X_train_norm    -   Normalizised training data attributes (X_train).
    X_test_norm     -   Normalized testing data attributes (X_test).
_____________________________________________________________________________________________________________________

ML hyperparameter description:
_______________________________________________________________________________

    MD      -   The maximum depth of the tree.
    NEST    -   The number of trees in the forest.
    Source: Scikit Learn 
    (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
_______________________________________________________________________________

CFEX hyperparameter description:
_______________________________________________________________________________

    INITIAL_NEURAL_ACTIVITY         -   Initial Neural Activity.
    EPSILON                         -   Noise Intensity.
    DISCRIMINATION_THRESHOLD        -   Discrimination Threshold.
    
    Source: Harikrishnan N.B., Nithin Nagaraj,
    When Noise meets Chaos: Stochastic Resonance in Neurochaos Learning,
    Neural Networks, Volume 143, 2021, Pages 425-435, ISSN 0893-6080,
    https://doi.org/10.1016/j.neunet.2021.06.025.
    (https://www.sciencedirect.com/science/article/pii/S0893608021002574)
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
length = 100

ar_dataset = load_pkl(f'path/Length_{length}/ar_dataset.pkl')
random_dataset = load_pkl(f'path/Length_{length}/rand_dataset.pkl')

# FFT of AR and Random datasets
freq_ar_dataset = np.abs(fftpack.fftfreq(length, 1)) # Discrete Fourier Transform sample frequencies
fft_ar_dataset = fftpack.fft(ar_dataset)
ampli_ar_dataset = 2/length * np.abs(fft_ar_dataset) 
freq_random_dataset = np.abs(fftpack.fftfreq(length, 1))
fft_random_dataset = fftpack.fft(random_dataset)
ampli_random_dataset = 2/length * np.abs(fft_random_dataset)

# Normalise the datasets
ampli_ar_dataset_scaled = scale_dataset(ampli_ar_dataset, ampli_ar_dataset.shape[0], ampli_ar_dataset.shape[1])
ampli_random_dataset_scaled = scale_dataset(ampli_random_dataset, ampli_random_dataset.shape[0], ampli_random_dataset.shape[1])

# Creating labels for AR and Random datasets: Random - 0, AR - 1
y_ar = np.ones((1250,1))
y_ran = np.zeros((1250,1))

# Reading data and labels from the dataset
X_val = np.concatenate((ampli_random_dataset_scaled, ampli_ar_dataset_scaled))
y_val = np.concatenate((y_ran,y_ar))

# Splitting the dataset for training, validation and testing (80-20)
X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_val,y_val,test_size=0.2, random_state=42)

# Validation
FOLD_NO = 5
INITIAL_NEURAL_ACTIVITY = [0.01]
DISCRIMINATION_THRESHOLD = [0.01]
EPSILON = [0.181] #np.arange(0.1, 0.49, 0.1) 0.161 0.01 0.01
BESTF1, BESTINA, BESTDT, BESTEPS, BESTNEST, BESTMD = k_cross_validation(FOLD_NO, X_train_val, y_train_val, X_test_val, y_test_val, INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON)

print("BEST F1SCORE", BESTF1)
print("BEST INITIAL NEURAL ACTIVITY = ", BESTINA)
print("BEST DISCRIMINATION THRESHOLD = ", BESTDT)
print("BEST EPSILON = ", BESTEPS)
print("BEST NEST = ", BESTNEST)
print("BEST MD= ", BESTMD)

# Reload the datasets to build the RFC model with best hyperparameters
X = np.concatenate((ampli_random_dataset, ampli_ar_dataset))
y = np.concatenate((y_ran,y_ar))

# Splitting the dataset for training, validation and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Extract CFEX features with best hyperparameters
X_train = extractCFEX(X_train, X_train.shape[0], X_train.shape[1], BESTINA, BESTEPS, BESTDT)
X_test = extractCFEX(X_test, X_test.shape[0], X_test.shape[1], BESTINA, BESTEPS, BESTDT)

# Build the RFC model with best hyperparameters
clf = RandomForestClassifier(n_estimators = BESTNEST, max_depth = BESTMD, random_state=42)
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
RESULT_PATH = PATH + f'/RFC-TUNING/AR_FT_CFEX_Hyperparameters/Length_{length}'
RESULT_PATH_FINAL = PATH + f'/TESTING-RESULTS/On_Training_set/AR_FT_CFEX_Results/Length_{length}'
RESULT_PATH_RFC = PATH + '/RFC-logs/AR_FT_CFEX_Checkpoints/'

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
    os.makedirs(RESULT_PATH_RFC)
except OSError:
    print ("Creation of the result directory %s not required" % RESULT_PATH_RFC)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH_RFC)

# Saving hyperparameters
np.save(RESULT_PATH+"/h_Q.npy", np.array([BESTINA])) 
np.save(RESULT_PATH+"/h_B.npy", np.array([BESTDT]))
np.save(RESULT_PATH+"/h_EPS.npy", np.array([BESTEPS]))
np.save(RESULT_PATH+"/h_MD.npy", np.array([BESTMD]))
np.save(RESULT_PATH+"/h_NEST.npy", np.array([BESTNEST]))

# Saving results
np.save(RESULT_PATH_FINAL+ "/Train_F1SCORE.npy", (F1SCORE_train))
np.save(RESULT_PATH_FINAL+ "/Test_F1SCORE.npy", (F1SCORE_test))

# Saving model
joblib.dump(clf, os.path.join(RESULT_PATH_RFC, f"rfc_{length}.pkl")) 