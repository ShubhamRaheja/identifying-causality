"""
Author: Shubham Raheja (shubhamraheja1999@gmail.com)
Code Description: A python code to tune the hyperparameters of TFR_LSTM on the AR dataset.
"""
import os
import numpy as np
import tensorflow as tf
import random
seed_value = 0
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(37)
random.seed(1254)
tf.random.set_seed(89)
from utils import load_pkl

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (f1_score, accuracy_score, classification_report)

'''
_______________________________________________________________________________

Rule used for renaming class labels: (Class Label - Numeric Code)
_______________________________________________________________________________

    AR series         -     0
    Random series     -     1
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

length = 75  # Pick length from 100, 75, 50

ar_dataset = load_pkl(f'path/Length_{length}/ar_dataset.pkl')
random_dataset = load_pkl(f'path/Length_{length}/rand_dataset.pkl')

# Creating labels for AR and Random datasets: Random - 0, AR - 1
y_ar = np.ones((1250,1))
y_ran = np.zeros((1250,1))

# Reading data and labels from the dataset
X = np.concatenate((random_dataset, ar_dataset))
y = np.concatenate((y_ran,y_ar))

# Binary matrix representation of the labels
y = to_categorical(y)

# Splitting the dataset for training, validation and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# Reshaping as tensor for LSTM algorithm.
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Checkpoints path 
PATH = os.path.dirname(__file__)
RESULT_PATH_TFR = PATH + '/TFR-logs/AR_LSTM_Checkpoints/'

try:
    os.makedirs(RESULT_PATH_TFR)
except OSError:
    print ("Creation of the result directory %s failed" % RESULT_PATH_TFR)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH_TFR)

# Algorithm - Transformer encoder / Building the model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    # x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    # x = layers.Dropout(dropout)(x)
    # x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LSTM(units=ff_dim, return_sequences=True)(x)
    return x + res

def model_builder(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
    learning_rate=1e-4,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(y_train.shape[1], activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics = ['accuracy'])
    checkpointer = ModelCheckpoint(filepath=RESULT_PATH_TFR + f"tfr_{length}_checkpoint.hdf5", verbose=1, monitor='loss', mode='min', save_best_only=True)
    model.fit(X_train,
                y_train,
                epochs=100,
                verbose=1,
                batch_size=32,
                # validation_split = 0.1,
                callbacks=[checkpointer],
                shuffle=True
                )
    return model


# Hyperparameters
input_shape = (X_train.shape[1],X_train.shape[2])
head_size = 128 #256
num_heads = 4
ff_dim = 10
num_transformer_blocks = 4
dropout = 0.25 
mlp_units = [64] #128
mlp_dropout = 0.4
learning_rate = 1e-4

# Build the Transformer encoder model with the best hyperparameters
model = model_builder(input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout,
    mlp_dropout,
    learning_rate
)
model.load_weights(RESULT_PATH_TFR + f"tfr_{length}_checkpoint.hdf5")

# Make predictions with trained model on train data
y_pred_traindata = np.argmax(model.predict(X_train), axis=-1)
y_train = np.argmax(y_train, axis=-1)
ACC = accuracy_score(y_train, y_pred_traindata)*100
F1SCORE_train = f1_score(y_train, y_pred_traindata, average="macro")
print('TRAIN: ACCURACY = ', ACC , " F1 SCORE = ", F1SCORE_train)
print(classification_report(y_train, y_pred_traindata))

y_pred_testdata = np.argmax(model.predict(X_test), axis=-1)
y_test = np.argmax(y_test, axis=-1)
ACC = accuracy_score(y_test, y_pred_testdata)*100
F1SCORE_test = f1_score(y_test, y_pred_testdata, average="macro")
print('TEST: ACCURACY = ', ACC , " F1 SCORE = ", F1SCORE_test)
print(classification_report(y_test, y_pred_testdata))


# Printing hyperparameters
print('Best Hyperparameters:')
print('Dimension of Q-K:', head_size)
print('Number of heads:', num_heads)
print('Dimension of FF layer:', ff_dim)
print('Number of transformer blocks:', num_transformer_blocks)
print('Dropout Rate (tfr):', dropout)
print('Number of mlp units:', mlp_units)
print('Dropout Rate (mlp):', mlp_dropout)
print('Learning Rate:', learning_rate)
# print('Best number of epochs:', best_epoch)

print("Saving Hyperparameter Tuning Results")

RESULT_PATH = PATH + f'/TFR-TUNING/AR_LSTM_Hyperparameters/Length_{length}'
RESULT_PATH_FINAL = PATH + f'/TESTING-RESULTS/On_Training_set/AR_LSTM_Results/Length_{length}'

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
np.save(RESULT_PATH+"/h_QK.npy", (head_size)) 
np.save(RESULT_PATH+"/h_Heads.npy", (num_heads)) 
np.save(RESULT_PATH+"/h_FFUnits.npy", (ff_dim))
np.save(RESULT_PATH+"/h_Blocks.npy", (num_transformer_blocks)) 
np.save(RESULT_PATH+"/h_DropoutTfr.npy", (dropout)) 
np.save(RESULT_PATH+"/h_MLPUnits.npy", (mlp_units))
np.save(RESULT_PATH+"/h_DropoutMLP.npy", (mlp_dropout))    
np.save(RESULT_PATH+"/h_LearningRate.npy", (learning_rate)) 


# Saving results
np.save(RESULT_PATH_FINAL+ "/Train_F1SCORE.npy", (F1SCORE_train))
np.save(RESULT_PATH_FINAL+ "/Test_F1SCORE.npy", (F1SCORE_test))