import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("data.csv")
# Reading dataset using pandas
x = dataset.iloc[:,:-1].values
# reading all rows of data of all col till last before
y = dataset.iloc[:,-1:].values
#all row val of last col


#Handling Missing Data
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
# Nan denotes the missing value and we calc mean for it. so strategy is mean
#to calc row mean, we keep axis as 1, for col it is 0
imputer = imputer.fit(x[:, 1:3])
# we have missing val in col 2,3 so we cal mean for 1,2 as index start from 0
x[:, 1:3] = imputer.transform(x[:, 1:3])


#Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoderX = LabelEncoder()
x[:, 0] = labelEncoderX.fit_transform(x[:, 0])
# specifies first col is to be labeled
oneHotEncoder = OneHotEncoder(categorical_features=[0])
x = oneHotEncoder.fit_transform(x).toarray()
# this creates a duclicate variables to categorize encoded variables
labelEncoderY = LabelEncoder()
y = labelEncoderY.fit_transform(y)


# Spliting data to training and testing
from sklearn.model_selection import train_test_split

xtrain, xtest, ytain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 0)
# specifies splitting of data to 20% of test data and rest to train data








