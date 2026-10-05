import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn


# Main function

# Command Line Input, Uncomment before submission
# print("File")
# data_file = input()
# data = pd.read_csv( train_file )

data = pd.read_csv( "Patient_Deterioration_10000.csv" )

# Data already standardized

# Split the data
train, test = sklearn.model_selection.train_test_split( data, test_size = 0.20)

# Learn the model

"""
      ___
    /'   `\
   /  o o  \
  |    ^    |
   \  '-'  /
    `-----'
   /|     |\
  / |     | \
     \___/
    /     \
   /_/   \_\

"""