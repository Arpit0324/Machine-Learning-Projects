import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pickle

placement_prediction= pd.read_csv("PlacementDataset.csv")

placement_prediction['salary'].fillna(value=0 , inplace = True )

placement_prediction.drop(['sl_no','ssc_b','hsc_b'], axis = 1 , inplace = True)

Q1 = placement_prediction['hsc_p'].quantile(0.25)
Q3 = placement_prediction['hsc_p'].quantile(0.75)
IQR = Q3 - Q1

filter = (placement_prediction['hsc_p'] >= Q1 - 1.5 * IQR) & (placement_prediction['hsc_p']<= Q3+ 1.5*IQR)
placement_filtered= placement_prediction.loc[filter]

placement_placed = placement_filtered[placement_filtered.salary!= 0]

#Label Encoding
from sklearn.preprocessing import LabelEncoder

object_cols= ['gender','workex','specialisation','status']

label_encoder = LabelEncoder()

for col in object_cols:
    placement_filtered[col]= label_encoder.fit_transform(placement_filtered[col])


# One Hot Encoding 
dummy_hsc_s = pd.get_dummies(placement_filtered['hsc_s'], prefix = 'dummy')
dummy_degree_t = pd.get_dummies(placement_filtered['degree_t'], prefix = 'dummy')

placement_coded = pd.concat([placement_filtered , dummy_hsc_s , dummy_degree_t],axis = 1)
placement_coded.drop(['hsc_s','degree_t','salary'],axis = 1 , inplace = True)


X = placement_coded.drop(['status'],axis=1)
y = placement_coded.status

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y , train_size = 0.8 , random_state = 1)

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logreg = LogisticRegression()

logreg.fit(X_train , y_train)

# y_pred = logreg.predict(X_test)

# print(logreg.score(X_test , y_test))

pickle.dump(logreg, open("model.pkl", "wb"))