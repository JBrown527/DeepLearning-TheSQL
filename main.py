# Team Members: Brodie O'Hara, Dyl Nebrida, Jonathan Brown
# Roles:
# DNN Design: Jonathan Brown
# DNN Implementation: Jonathan Brown, Dyl Nebrida
# Data Partitioning and Standardization: Jonathan Brown, Dyl Nebrida, Brodie O'Hera
# Hyperparameter Search: Brodie O'Hera
# Experimentation and Evaluation: Jonathan Brown, Dyl Nebrida, Brodie O'Hera

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
from sklearn.metrics import log_loss, confusion_matrix
from sklearn.preprocessing import StandardScaler

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()

# Main function

# Command Line Input, Uncomment before submission
print("File")
data_file = input()
data = pd.read_csv( data_file )

# Temporary, comment out / delete before submitting
# data = pd.read_csv( "Patient_Deterioration_10000.csv" )

# Separate features and target, converting both to float32
X = data.drop(columns=['Deterioration_Next_6h']).to_numpy(dtype=np.float32)
y = data['Deterioration_Next_6h'].to_numpy(dtype=np.float32)

# Split into remaining data and test data
X_remaining, test, y_remaining, test_target = sklearn.model_selection.train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=0
)

# Split remaining data into training and validation
train, validation, train_target, validation_target = sklearn.model_selection.train_test_split(
    X_remaining,
    y_remaining,
    test_size=0.20,
    random_state=0
)

# Standardize features
scaler = StandardScaler()

train = scaler.fit_transform(train).astype(np.float32)
validation = scaler.transform(validation).astype(np.float32)
test = scaler.transform(test).astype(np.float32)

# Learn the model

results = []
best_model = None
best_accuracy = -1
best_loss = float('inf')
best_params = None
baseline_accuracy = None

# For each hyper parameter value we want to test
for l_two in [ 0.0, 0.003, 0.01, 0.03 ]:
   for dropout in [0.0, 0.2, 0.5]:
      for standardDeviation in [ 0.0, 0.05, 0.15 ]:
         tf.keras.backend.clear_session()
         tf.keras.utils.set_random_seed(0)

         ann = tf.keras.models.Sequential()
         ann.add(tf.keras.layers.Input(
            shape=(768,)))
         ann.add(tf.keras.layers.GaussianNoise(stddev = standardDeviation, seed=0))
         ann.add(tf.keras.layers.Dense(
            units=513, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.L2(l_two)))
         ann.add(tf.keras.layers.Dropout(dropout, seed=0))
         ann.add(tf.keras.layers.Dense(
            units=513, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.L2(l_two)))
         ann.add(tf.keras.layers.Dropout(dropout, seed=0))
         ann.add(tf.keras.layers.Dense(
            units=1, 
            activation='sigmoid',
            kernel_regularizer=tf.keras.regularizers.L2(l_two)))

         ann.compile( 
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['Accuracy']
         )

         ann.fit(
            train,
            train_target,
            batch_size=64,
            epochs=100,
            verbose=0
         )

         probabilities = ann.predict(validation, verbose=0)
         
         y_pred = (probabilities > 0.5).astype(int).flatten()

         accuracy = (y_pred == validation_target).mean()

         validation_loss = log_loss(
            validation_target,
            probabilities.flatten(),
            labels=[0, 1]
         )
         
         if best_model is None:
            better = True

         elif accuracy > best_accuracy:
            better = True

         elif accuracy == best_accuracy:
            if validation_loss < best_loss:
               better = True

            elif validation_loss == best_loss:
               if (l_two, dropout, standardDeviation) < best_params:
                     better = True
               else:
                     better = False

            else:
               better = False

         else:
            better = False

         if better:
            best_model = ann
            best_accuracy = accuracy
            best_loss = validation_loss
            best_params = (l_two, dropout, standardDeviation)
            
         if l_two == 0.0 and dropout == 0.0 and standardDeviation == 0.0:
            baseline_probabilities = ann.predict(test, verbose=0)

            baseline_predictions = (baseline_probabilities > 0.5).astype(int).flatten()

            baseline_accuracy = (baseline_predictions == test_target).mean()
                  
         results.append([l_two, dropout, standardDeviation, accuracy * 100])

# Determine which set of hyperparameters is best using the *validation* set. 
# (Accuracy, of course, uses the test set)


# Create (replace if exists) the excel file called "Project2_Results.xlsx", 
# one workbook called "Search Results"

results_df = pd.DataFrame(
    results,
    columns=[
        "L2 strength",
        "Dropout rate",
        "Gaussian noise standard deviation",
        "Validation accuracy (%)"
    ]
)

# output excel file, also enforce accuracy decimal digits
with pd.ExcelWriter("Project2_Results.xlsx", engine="xlsxwriter") as writer:
    results_df.to_excel(writer, sheet_name="Search Results", index=False)

    worksheet = writer.sheets["Search Results"]
    number_format = writer.book.add_format({"num_format": "0.00"})

    worksheet.set_column(3, 3, 25, number_format)


# use the best model, get the stats
probabilities = best_model.predict(test, verbose=0)

predictions = (probabilities > 0.5).astype(int).flatten()

test_accuracy = (predictions == test_target).mean()

cm = confusion_matrix(test_target, predictions, labels=[0,1])

# Print baseline accuracy, selected DNN test accuracy, selected DNN confusion matrix
print(f'Baseline Test Accuracy: {baseline_accuracy * 100:.2f}%')
print(f'Selected DNN Test Accuracy: {test_accuracy * 100:.2f}%')

print("Selected DNN Confusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()

print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")