"""
Author: Shubham Raheja (shubhamraheja1999@gmail.com)
Code Description: A python code to tune the hyperparameters of RFC on the AR dataset.
"""
import os
import numpy as np
import pickle
import joblib
from utils import load_pkl

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, accuracy_score, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

'''
_______________________________________________________________________________

Rule used for naming class labels: (Class Label - Numeric Code)
_______________________________________________________________________________

    AR series         -     0
    Random series     -     1
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
length = 50

ar_dataset = load_pkl(f'path/Length_{length}/ar_dataset.pkl')
random_dataset = load_pkl(f'path/Length_{length}/rand_dataset.pkl')

# Creating labels for AR and Random datasets: Random - 0, AR - 1
y_ar = np.ones((1250,1))
y_ran = np.zeros((1250,1))

# Reading data and labels from the dataset
X = np.concatenate((random_dataset, ar_dataset))
y = np.concatenate((y_ran,y_ar))

# Splitting the dataset for training, validation and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# Algorithm - Random Forest Classifier
n_estimator = [1, 10, 100, 1000]
BESTF1 = 0
FOLD_NO = 5
KF = KFold(n_splits= FOLD_NO, random_state=42, shuffle=True)  
KF.get_n_splits(X_train) 
print(KF) 
for NEST in n_estimator:
                
    for MD in range(1,6):
        
    
        FSCORE_TEMP=[]
    
        for TRAIN_INDEX, VAL_INDEX in KF.split(X_train):
            
            X_TRAIN, X_VAL = X_train[TRAIN_INDEX], X_train[VAL_INDEX]
            Y_TRAIN, Y_VAL = y_train[TRAIN_INDEX], y_train[VAL_INDEX]
        
            
            clf = RandomForestClassifier(n_estimators = NEST, max_depth = MD, random_state=42)
            clf.fit(X_TRAIN, Y_TRAIN.ravel())
            Y_PRED = clf.predict(X_VAL)
            f1 = f1_score(Y_VAL, Y_PRED, average='macro')
            FSCORE_TEMP.append(f1)
            print('F1 Score', f1)
        print("Mean F1-Score for N-EST = ", NEST," MD = ", MD," is  = ",  np.mean(FSCORE_TEMP)  )
        if(np.mean(FSCORE_TEMP) > BESTF1):
            BESTF1 = np.mean(FSCORE_TEMP)
            BESTNEST = NEST
            BESTMD = MD

        
print("BEST F1SCORE", BESTF1)
print("BEST MD = ", BESTMD)
print("BEST NEST = ", BESTNEST)


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
RESULT_PATH = PATH + f'/RFC-TUNING/AR_Hyperparameters/Length_{length}'
RESULT_PATH_FINAL = PATH + f'/TESTING-RESULTS/On_Training_set/AR_Results/Length_{length}'
RESULT_PATH_RFC = PATH + '/RFC-logs/AR_Checkpoints/'

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
np.save(RESULT_PATH+"/h_MD.npy", np.array([BESTMD]))
np.save(RESULT_PATH+"/h_NEST.npy", np.array([BESTNEST]))  

# Saving results
np.save(RESULT_PATH_FINAL+ "/Train_F1SCORE.npy", (F1SCORE_train))
np.save(RESULT_PATH_FINAL+ "/Test_F1SCORE.npy", (F1SCORE_test))

# Saving model
joblib.dump(clf, os.path.join(RESULT_PATH_RFC, f"rfc_{length}.pkl")) 

