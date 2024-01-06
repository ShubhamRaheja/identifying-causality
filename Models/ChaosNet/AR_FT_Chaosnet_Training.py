"""
Author: Shubham Raheja (shubhamraheja1999@gmail.com)
Code Description: A python code to tune the hyperparameters of ChaosNet on the AR+FT dataset.
"""
import os
import numpy as np
from scipy import fft as fftpack
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, accuracy_score, classification_report)
from Codes import k_cross_validation, chaosnet
from utils import load_pkl, scale_dataset, extractCFEX
import ChaosFEX.feature_extractor as CFEX

'''
_______________________________________________________________________________

Rule used for renaming class labels: (Class Label - Numeric Code)
_______________________________________________________________________________

    AR+FT series         -     0
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
y_ar = np.ones((ar_dataset.shape[0],1))
y_ran = np.zeros((random_dataset.shape[0],1))

# Reading data and labels from the dataset
X_val = np.concatenate((ampli_random_dataset_scaled, ampli_ar_dataset_scaled))
y_val = np.concatenate((y_ran, y_ar))


# Splitting the dataset for training, validation and testing (80-20)
X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_val, y_val, test_size=0.2, stratify=y_val, random_state=42)

# Validation
FOLD_NO = 5
INITIAL_NEURAL_ACTIVITY = np.arange(0.01, 0.25, 0.01)
DISCRIMINATION_THRESHOLD = np.arange(0.01, 0.25, 0.01) #Mean F1-Score for Q =  0.15000000000000002 B =  0.25 EPSILON =  0.241  is  =  0.9438435827589835
EPSILON = np.arange(0.001, 0.499, 0.01)
BESTF1, BESTINA, BESTDT, BESTEPS = k_cross_validation(FOLD_NO, X_train_val, y_train_val, X_test_val, y_val, INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON)

print("BEST F1SCORE", BESTF1)
print("BEST INITIAL NEURAL ACTIVITY = ", BESTINA)
print("BEST DISCRIMINATION THRESHOLD = ", BESTDT)
print("BEST EPSILON = ", BESTEPS)

CFEX_ar_dataset = CFEX.transform(ampli_ar_dataset_scaled, BESTINA[0], 10000, BESTEPS[0], BESTDT[0])[:,0:length]
CFEX_random_dataset = CFEX.transform(ampli_random_dataset_scaled, BESTINA[0], 10000, BESTEPS[0], BESTDT[0])[:,0:length]

# Reload the datasets to build the ChaosNet model with best hyperparameters
X = np.concatenate((CFEX_random_dataset, CFEX_ar_dataset))
y = np.concatenate((y_ran,y_ar))

# Splitting the dataset for training, validation and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Make predictions with trained model on train data        
mean_each_class, y_pred_testdata = chaosnet(X_train, y_train, X_test)
ACC = accuracy_score(y_test, y_pred_testdata)*100
F1SCORE_test = f1_score(y_test, y_pred_testdata, average='macro')
print('TEST: ACCURACY = ', ACC , " F1 SCORE = ", F1SCORE_test)
print(classification_report(y_test, y_pred_testdata))

print("Saving Hyperparameter Tuning Results")

PATH = os.path.dirname(__file__)
RESULT_PATH = PATH + f'/CFEX-TUNING/AR_FT_Hyperparameters/Length_{length}'
RESULT_PATH_FINAL = PATH + f'/TESTING-RESULTS/On_Training_set/AR_FT_Results/Length_{length}'

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
    

# Saving hyperparameters
np.save(RESULT_PATH+"/h_Q.npy", np.array([BESTINA])) 
np.save(RESULT_PATH+"/h_B.npy", np.array([BESTDT]))
np.save(RESULT_PATH+"/h_EPS.npy", np.array([BESTEPS]))

# Saving results
np.save(RESULT_PATH_FINAL+ "/Test_F1SCORE.npy", (F1SCORE_test))


