# -*- coding: utf-8 -*-
"""
Author: Harikrishnan NB
Dtd: 22 Dec. 2020
Edited by Shubham Raheja (shubhamraheja1999@gmail.com) to fit the causality identification experiments.
ChaosNet decision function
"""

import os
seed_value = 0
os.environ['PYTHONHASHSEED'] = str(seed_value)
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import ChaosFEX.feature_extractor as CFX
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
                    
      
def k_cross_validation(FOLD_NO, traindata, trainlabel, testdata, testlabel, INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON):
    """

    Parameters
    ----------
    FOLD_NO : TYPE-Integer
        DESCRIPTION-K fold classification.
    traindata : TYPE-numpy 2D array
        DESCRIPTION - Traindata
    trainlabel : TYPE-numpy 2D array
        DESCRIPTION - Trainlabel
    testdata : TYPE-numpy 2D array
        DESCRIPTION - Testdata
    testlabel : TYPE - numpy 2D array
        DESCRIPTION - Testlabel
    INITIAL_NEURAL_ACTIVITY : TYPE - numpy 1D array
        DESCRIPTION - initial value of the chaotic skew tent map.
    DISCRIMINATION_THRESHOLD : numpy 1D array
        DESCRIPTION - thresholds of the chaotic map
    EPSILON : TYPE numpy 1D array
        DESCRIPTION - noise intenity for NL to work (low value of epsilon implies low noise )

    Returns
    -------
    FSCORE, Q, B, EPS, EPSILON

    """
    # trainlabel = to_categorical(trainlabel)
    BESTF1 = 0
    KF = KFold(n_splits= FOLD_NO, random_state=42, shuffle=True)  
    KF.get_n_splits(traindata) 
    print(KF) 
    
    
    for DT in DISCRIMINATION_THRESHOLD:
        
        for INA in INITIAL_NEURAL_ACTIVITY:
            
            for EPSILON_1 in EPSILON:
                FSCORE_TEMP=[]
                    
                for TRAIN_INDEX, VAL_INDEX in KF.split(traindata):
                    
                    X_TRAIN, X_VAL = traindata[TRAIN_INDEX], traindata[VAL_INDEX]
                    Y_TRAIN, Y_VAL = trainlabel[TRAIN_INDEX], trainlabel[VAL_INDEX]

                    # Extract CFX features
                    FEATURE_MATRIX_TRAIN = CFX.transform(X_TRAIN, INA, 10000, EPSILON_1, DT)
                    FEATURE_MATRIX_TRAIN = FEATURE_MATRIX_TRAIN[:,0:X_TRAIN.shape[1]]
                    print(f'Shape of X_train TTSS feature matrix: {FEATURE_MATRIX_TRAIN.shape}')
                    FEATURE_MATRIX_VAL = CFX.transform(X_VAL, INA, 10000, EPSILON_1, DT)
                    FEATURE_MATRIX_VAL = FEATURE_MATRIX_VAL[:,0:X_TRAIN.shape[1]]
                    print(f'Shape of X_val TTSS feature matrix: {FEATURE_MATRIX_VAL.shape}')

                    # Reshaping as tensor for LSTM algorithm.            
                    FEATURE_MATRIX_TRAIN = np.reshape(FEATURE_MATRIX_TRAIN,(FEATURE_MATRIX_TRAIN.shape[0], 1, FEATURE_MATRIX_TRAIN.shape[1]))
                    FEATURE_MATRIX_VAL = np.reshape(FEATURE_MATRIX_VAL,(FEATURE_MATRIX_VAL.shape[0], 1, FEATURE_MATRIX_VAL.shape[1]))
                    def model_builder(lstm_units, dropout_rate, dense_units, learning_rate):
                        model = Sequential()            
                        model.add(LSTM(units=lstm_units, input_shape=(FEATURE_MATRIX_TRAIN.shape[1],FEATURE_MATRIX_TRAIN.shape[2]))) # Defining the number of LSTM units and input shape
                        model.add(Dropout(dropout_rate)) # Defining the dropout rate
                        model.add(Dense(units=dense_units,activation='relu')) # Defining the number of dense units
                        model.add(Dense(Y_TRAIN.shape[1], activation='softmax')) 
                        model.compile(loss='categorical_crossentropy', 
                                    optimizer=Adam(learning_rate=learning_rate),
                                    metrics = ['accuracy'])
                        model.fit(FEATURE_MATRIX_TRAIN,
                                        Y_TRAIN,
                                        epochs = 50,
                                        verbose=1,
                                        batch_size=8,
                                        # validation_split = 0.1,
                                        shuffle = True
                                        )
                        return model

                    # Hyperparameters
                    lstm_units = 10
                    dropout_rate = 0.5
                    dense_units = 10
                    learning_rate = 0.01

                    # Build the LSTM model with the best hyperparameters
                    model = model_builder(lstm_units, dropout_rate, dense_units, learning_rate)
                    Y_PRED = np.argmax(model.predict(FEATURE_MATRIX_VAL), axis=-1)
                    Y_VAL_true = np.argmax(Y_VAL, axis=-1)
                    F1SCORE = f1_score(Y_VAL_true, Y_PRED, average="macro")
                    FSCORE_TEMP.append(F1SCORE)
                    print(F1SCORE)
                print("Mean F1-Score for Q = ", INA,"B = ", DT,"EPSILON = ", EPSILON_1," is  = ",  np.mean(FSCORE_TEMP)  )

                if(np.mean(FSCORE_TEMP) > BESTF1):
                    BESTF1 = np.mean(FSCORE_TEMP)
                    BESTINA = INA
                    BESTDT = DT
                    BESTEPS = EPSILON_1
    return [BESTF1, BESTINA, BESTDT, BESTEPS]


    





