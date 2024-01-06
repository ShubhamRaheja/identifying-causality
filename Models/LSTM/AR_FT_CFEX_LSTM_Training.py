"""
Author: Shubham Raheja (shubhamraheja1999@gmail.com)
Code Description: A python code to tune the hyperparameters of LSTM on the AR+FT+CFEX dataset.
"""
import os
seed_value = 0
os.environ['PYTHONHASHSEED'] = str(seed_value)
import numpy as np
import tensorflow as tf
import random
import pickle
from scipy import fft as fftpack
np.random.seed(37)
random.seed(1254)
tf.random.set_seed(89)
from utils import load_pkl, scale_dataset, extractCFEX
from Codes import k_cross_validation

from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (f1_score, accuracy_score, classification_report)

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
_______________________________________________________________________________

DL hyperparameter description:
_______________________________________________________________________________

    LSTM units - Positive integer, dimensionality of the output space. 
    Source : Keras 
    (https://keras.io/api/layers/recurrent_layers/lstm/)
    
    Dense layer units - Positive integer, dimensionality of the output space.
    Source : Keras 
    (https://keras.io/api/layers/core_layers/dense/)
    
    Dense layer activation - Activation function of the dense layer.
    Source : Keras
    (https://keras.io/api/layers/core_layers/dense/)
    
    Dropout rate - Float between 0 and 1. Fraction of the input units to drop.
    The dropout layer randomly sets input units to 0 with a frequency of rate 
    at each step during training time, which prevents overfitting. 
    Source : Keras
    (https://keras.io/api/layers/regularization_layers/dropout/)
    
    Learning rate - The learning rate is a hyperparameter that controls how 
    much to change the model in response to the estimated error each time the 
    odel weights are updated.
    
    Here the learning rate is being set for the Adam optimizer. 
    
    Adam optimization is a stochastic gradient descent method 
    that is based on adaptive estimation of first-order and second-order 
    moments.
    Source : Keras
    (https://keras.io/api/optimizers/adam/)
    
    
    The above mentioned hyperparameters are tuned using KerasTuner. It is a 
    general-purpose hyperparameter tuning library.
    Source : Keras
    (https://keras.io/guides/keras_tuner/)
    
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
    equal; 'macro' calculates metrics for each label, and finds their 
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

# Normalise the datasets
ampli_ar_dataset_scaled = scale_dataset(ampli_ar_dataset, ampli_ar_dataset.shape[0], ampli_ar_dataset.shape[1])
ampli_random_dataset_scaled = scale_dataset(ampli_random_dataset, ampli_random_dataset.shape[0], ampli_random_dataset.shape[1])

# Creating labels for AR and Random datasets: Random - 0, AR - 1
y_ar = np.ones((1250, 1))
y_ran = np.zeros((1250, 1))

# Reading data and labels from the dataset
X_val = np.concatenate((ampli_random_dataset_scaled, ampli_ar_dataset_scaled))
y_val = np.concatenate((y_ran, y_ar))

# Binary matrix representation of the labels
y_val = to_categorical(y_val)

# Splitting the dataset for training, validation and testing (80-20)
X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_val,y_val,test_size=0.2, stratify=y_val, random_state=42)

# Validation
FOLD_NO = 5
INITIAL_NEURAL_ACTIVITY = [0.02] #, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.09] #np.arange(0.01, 0.98, 0.01) #[0.21]
DISCRIMINATION_THRESHOLD =[0.01] #, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
EPSILON = [0.161] #, 0.181 0.171, 0.161, 0.151, 0.141, 0.131, 0.121, 0.111, 0.101, 0.101] # np.arange(0.1, 0.49, 0.1)
BESTF1, BESTINA, BESTDT, BESTEPS = k_cross_validation(FOLD_NO, X_train_val, y_train_val, X_test_val, y_test_val, INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON)

print("BEST F1SCORE", BESTF1)
print("BEST INITIAL NEURAL ACTIVITY = ", BESTINA)
print("BEST DISCRIMINATION THRESHOLD = ", BESTDT)
print("BEST EPSILON = ", BESTEPS)

# Reload the datasets to build the LSTM model with best hyperparameters
X = np.concatenate((ampli_random_dataset, ampli_ar_dataset))
y = np.concatenate((y_ran,y_ar))

# Binary matrix representation of the labels
y = to_categorical(y)

# Splitting the dataset for training, validation and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Extract CFEX features with best hyperparameters
X_train = extractCFEX(X_train, X_train.shape[0], X_train.shape[1], BESTINA, BESTEPS, BESTDT)
X_test = extractCFEX(X_test, X_test.shape[0], X_test.shape[1], BESTINA, BESTEPS, BESTDT)

# Reshaping as tensor for LSTM algorithm.
X_train = np.reshape(X_train ,(X_train.shape[0], X_train.shape[1], 1)) 
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1], 1))

# Checkpoints path 
PATH = os.getcwd()
RESULT_PATH_DL = PATH + '/Models/LSTM/LSTM-logs/AR_FT_CFEX_Checkpoints/'

try:
    os.makedirs(RESULT_PATH_DL)
except OSError:
    print ("Creation of the result directory %s failed" % RESULT_PATH_DL)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH_DL)

# Algorithm - LSTM / Building the model

def model_builder(lstm_units, dropout_rate, dense_units, learning_rate):
    model = Sequential()            
    model.add(LSTM(units=lstm_units, input_shape=(X_train.shape[1],X_train.shape[2]))) # Defining the number of LSTM units and input shape
    model.add(Dropout(dropout_rate)) # Defining the dropout rate
    model.add(Dense(units=dense_units,activation='relu')) # Defining the number of dense units
    model.add(Dense(y_train.shape[1], activation='softmax')) 
    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics = ['accuracy'])
    checkpointer = ModelCheckpoint(filepath=RESULT_PATH_DL + f"lstm_{length}_checkpoint.hdf5", verbose=1, monitor='loss', mode='min', save_best_only=True)
    model.fit(X_train,
                    y_train,
                    epochs=100,
                    verbose=1,
                    batch_size=8,
                    # validation_split = 0.1,
                    callbacks=[checkpointer],
                    shuffle=True
                    )
    return model


# Hyperparameters
lstm_units = 10
dropout_rate = 0.4
dense_units = 10
learning_rate = 0.0001

# Build the LSTM model with the best hyperparameters
model = model_builder(lstm_units, dropout_rate, dense_units, learning_rate)
model.load_weights(RESULT_PATH_DL + f"lstm_{length}_checkpoint.hdf5")

# Make predictions with trained model on train data
y_pred_traindata = np.argmax(model.predict(X_train), axis=1)
y_train= np.argmax(y_train,axis=1)
ACC = accuracy_score(y_train, y_pred_traindata)*100
F1SCORE_train = f1_score(y_train, y_pred_traindata, average="macro")
print('TRAIN: ACCURACY = ', ACC , " F1 SCORE = ", F1SCORE_train)
print(classification_report(y_train, y_pred_traindata))

y_pred_testdata = np.argmax(model.predict(X_test), axis=1)
y_test= np.argmax(y_test,axis=1)
ACC = accuracy_score(y_test, y_pred_testdata)*100
F1SCORE_test = f1_score(y_test, y_pred_testdata, average="macro")
print('TEST: ACCURACY = ', ACC , " F1 SCORE = ", F1SCORE_test)
print(classification_report(y_test, y_pred_testdata))


# Printing hyperparameters
print('Best Hyperparameters:')
print('LSTM Units:', lstm_units)
print('Dense Layer Units:', dense_units)
print('Dropout Rate:', dropout_rate)
print('Learning Rate:', learning_rate)
# print('Best number of epochs:', best_epoch)

print("Saving Hyperparameter Tuning Results")

PATH = os.getcwd()
RESULT_PATH = PATH + f'/Models/LSTM/LSTM-TUNING/AR_FT_CFEX_Hyperparameters/Length_{length}'
RESULT_PATH_FINAL = PATH + f'/Models/LSTM/TESTING-RESULTS/On_Training_set/AR_FT_CFEX_Results/Length_{length}'

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
np.save(RESULT_PATH+"/h_Units.npy", (lstm_units)) 
np.save(RESULT_PATH+"/h_Dense.npy", (dense_units)) 
np.save(RESULT_PATH+"/h_DropoutRate.npy", (dropout_rate)) 
np.save(RESULT_PATH+"/h_LearningRate.npy", (learning_rate)) 
np.save(RESULT_PATH+"/h_Q.npy", np.array([BESTINA])) 
np.save(RESULT_PATH+"/h_B.npy", np.array([BESTDT]))
np.save(RESULT_PATH+"/h_EPS.npy", np.array([BESTEPS]))
# np.save(RESULT_PATH+"/h_BestEpoch.npy", best_epoch) 
np.save(RESULT_PATH_FINAL+ "/Train_F1SCORE.npy", (F1SCORE_train))
np.save(RESULT_PATH_FINAL+ "/Test_F1SCORE.npy", (F1SCORE_test))