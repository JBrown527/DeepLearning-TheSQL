import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.preprocessing import StandardScaler
import uwuify

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()

# Main function

# Command Line Input, Uncomment before submission
# print("File")
# data_file = input()
# data = pd.read_csv( train_file )

# Temporary, comment out / delete before submitting
data = pd.read_csv( "Patient_Deterioration_10000.csv" )

# Data already standardized

# Split the data
remaining, test = sklearn.model_selection.train_test_split( 
   data, 
   test_size = 0.20, 
   random_state= 0
)
train, validation = sklearn.model_selection.train_test_split( 
   remaining, 
   test_size = 0.20, 
   random_state= 0
)

train_target = train['Deterioration_Next_6h']
validation_target = validation['Deterioration_Next_6h']
test_target = test['Deterioration_Next_6h']

train = train.drop(columns=['Deterioration_Next_6h'])
validation = validation.drop(columns=['Deterioration_Next_6h'])
test = test.drop(columns=['Deterioration_Next_6h'])

scaler = StandardScaler()
train = scaler.fit_transform( train )
validation = scaler.transform( validation )
test = scaler.transform( test )

train = train.astype( np.float32 )
validation = validation.astype( np.float32 )
test = test.astype( np.float32 )

print(uwuify.uwu("Let's train our test set!"))

# Learn the model

l_two = [ 0.0, 0.003, 0.01, 0.03 ]
dropout = [0.0, 0.2, 0.5]
standardDeviation = [ 0.0, 0.05, 0.15 ]

# For each hyper parameter value we want to test
for i in range (1, 4):
   for j in range ( 1, 3 ):
      for k in range ( 1, 3 ):
         tf.keras.backend.clear_session()
         tf.keras.utils.set_random_seed(0)

         ann = tf.keras.models.Sequential()
         ann.add(tf.keras.layers.Dense(
            units=768, 
            activiation='relu',
            kernel_regularizer=tf.keras.regularizers.L2(l_two[ i ])))
         ann.add(tf.kera.layers.GassianNoise(stddev = standardDeviation[ k ], seed=0))
         ann.add(tf.keras.layers.Dense(
            units=513, 
            activiation='relu',
            kernel_regularizer=tf.keras.regularizers.L2(l_two[ i ])))
         ann.add(tf.keras.layers.Dropout(dropout[ j ], seed=0))
         ann.add(tf.keras.layers.Dense(
            units=513, 
            activiation='relu',
            kernel_regularizer=tf.keras.regularizers.L2(l_two[ i ])))
         ann.add(tf.keras.layers.Dropout(dropout[ j ], seed=0))
         ann.add(tf.keras.layers.Dense(
            units=1, 
            activiation='sigmoid',
            kernel_regularizer=tf.keras.regularizers.L2(l_two[ i ])))

         ann.compile( 
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['Accuracy']
         )

         ann.fit(
            train,
            train_target,
            batch_size=64,
            epochs=10,
            verbose=0
         )

         probabilities = ann.predict(test, verbose=0)
         y_pred = (probabilities > 0.5).astype(int).flatten()

         accuracy = (y_pred == test_target.to_numpy()).mean()

# Determine which set of hyperparameters is best using the *validation* set. 
# (Accuracy, of course, uses the test set)

# Create (replace if exists) the excel file called "Project2_Results.xlsx", 
# one workbook called "Search Results"

# First row column headings: L2 strength, Dropout rate, 
# Gaussian noise standard deviation, Validation accuracy (%)

# Create 36 rows, one for each combination
# Accuracy desplayed as number between 0 and 100 (2 decimal places)

# Print baseline accuracy, selected DNN test accuracy, selected DNN confusion matrix



"""
      ___
    /'   `\
   /  o o  \
  |    ^    |
   \  '-'  /
    '-----'
   /|     |\
  / |     | \
     \___/
    /     \
   /_/   \_\

"""