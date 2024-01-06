# -*- coding: utf-8 -*-
"""
Author: Harikrishnan NB
Dtd: 22 Dec. 2020
Edited by Shubham Raheja (shubhamraheja1999@gmail.com) to fit the causality identification experiments.
ChaosNet decision function
"""

import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import ChaosFEX.feature_extractor as CFEX
from sklearn.ensemble import RandomForestClassifier



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
    
    BESTF1 = 0
    KF = KFold(n_splits= FOLD_NO, random_state=42, shuffle=True) 
    KF.get_n_splits(traindata) 
    print(KF) 
    
    n_estimator = [1, 10, 100, 1000, 10000]



    for DT in DISCRIMINATION_THRESHOLD:
        
        for INA in INITIAL_NEURAL_ACTIVITY:
            
            for EPSILON_1 in EPSILON:
                
                for NEST in n_estimator:
                
                    for MD in range(1,6):   #11):  
                        FSCORE_TEMP=[]
                                                                 
                        for TRAIN_INDEX, VAL_INDEX in KF.split(traindata):
                            
                            X_TRAIN, X_VAL = traindata[TRAIN_INDEX], traindata[VAL_INDEX]
                            Y_TRAIN, Y_VAL = trainlabel[TRAIN_INDEX], trainlabel[VAL_INDEX]
                
        
                            # Extract features  
                            FEATURE_MATRIX_TRAIN = CFEX.transform(X_TRAIN, INA, 10000, EPSILON_1, DT)
                            FEATURE_MATRIX_TRAIN = FEATURE_MATRIX_TRAIN[:,0:X_TRAIN.shape[1]]
                            print(f'Shape of X_train TTSS feature matrix: {FEATURE_MATRIX_TRAIN.shape}')
                            FEATURE_MATRIX_VAL = CFEX.transform(X_VAL, INA, 10000, EPSILON_1, DT)    
                            FEATURE_MATRIX_VAL = FEATURE_MATRIX_VAL[:,0:X_TRAIN.shape[1]]
                            print(f'Shape of X_val TTSS feature matrix: {FEATURE_MATRIX_VAL.shape}') 
                            clf = RandomForestClassifier( n_estimators = NEST, max_depth = MD, random_state=42)
                            clf.fit(FEATURE_MATRIX_TRAIN, Y_TRAIN.ravel())
                            Y_PRED = clf.predict(FEATURE_MATRIX_VAL)
                            F1SCORE = f1_score(Y_VAL, Y_PRED, average="macro")
                            FSCORE_TEMP.append(F1SCORE)
                        print("Mean F1-Score for Q = ", INA,"B = ", DT,"EPSILON = ", EPSILON_1," is  = ",  np.mean(FSCORE_TEMP)  )
        
                        if(np.mean(FSCORE_TEMP) > BESTF1):
                            BESTF1 = np.mean(FSCORE_TEMP)
                            BESTINA = INA
                            BESTDT = DT
                            BESTEPS = EPSILON_1
                            BESTNEST = NEST
                            BESTMD = MD

    return [BESTF1, BESTINA, BESTDT, BESTEPS, BESTNEST, BESTMD]                                                  
                