# -*- coding: utf-8 -*-
"""Logisitic_regression

Original file is located at
    https://colab.research.google.com/drive/1DjdYEXO3H7FOmTAQRedyJQkqDMsG_FiR

## Data_Preprocessing
"""

pip install pandas

import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
df=pd.read_csv("/content/framingham.csv")

df

missing_val=["heartRate","glucose"]
for i in missing_val:
  mean_val=df[i].mean()
  df[i].fillna(mean_val,inplace=True)

x=df.iloc[:,[14]].values
y=df.iloc[:,[15]].values

a=df["TenYearCHD"].value_counts()[0]
b=df["TenYearCHD"].value_counts()[1]
print(a,b)

#independent variable
x

#Dependent
y
type(y)

#splitting train and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

x_train

"""## Feature Scaling"""

from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)

"""##Model Fitting"""

from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

"""# Prediction"""

y_pred=classifier.predict(x_test)

y_pred

"""## Testing"""

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)

#confusion matrix
cm
