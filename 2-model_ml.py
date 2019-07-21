# MODEL MACHINE LEARNING

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
warnings.filterwarnings("ignore")

#==================================================== LOAD DATA ==============================================================================================================================================

df = pd.read_csv('train.csv')

dfx = df[['ram','fc','pc', 'battery_power', 'px_height','px_width']]
dfy = df['price_range']

# print(dfx.head())
# print(dfx['fc'].unique())
# print(dfx['pc'].unique())
# print(dfx['px_height'].max())
# print(dfx['px_width'].max())
# print(dfx['battery_power'].max())
print(df.columns)

#==================================================== SPLITTING DATA ==============================================================================================================================================

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(
    dfx, dfy, test_size= 0.1, random_state = 10
)

#=========================================================== LOGISTIC REGRESSION ==============================================================================================================================================

from sklearn.linear_model import LogisticRegression
model_LGR = LogisticRegression()
model_LGR.fit(xtrain, ytrain)

# accuracy
acc_LGR = model_LGR.score(xtest,ytest)
print('Accuracy LGR: ', acc_LGR * 100, '%')                         # Accuracy LGR: 71  %
# print(model_LGR.coef_)

# prediction of best price (Logistic Regression)
pred_LGR = model_LGR.predict(xtest)
print('prediction LGR (y prediction): ', pred_LGR[0])
print('y real value: ', ytest.iloc[0])

# ====================================================   RANDOM FOREST CLASSIFIER ==============================================================================================================================================

from sklearn.ensemble import RandomForestClassifier
model_RF = RandomForestClassifier(n_estimators=1000)
model_RF.fit(xtrain,ytrain)

# accuracy
acc_RF = model_RF.score(xtest,ytest)
print('Accuracy RF: ', acc_RF * 100, '%')                         # Accuracy RF: 72 %
# print(model_RF.feature_importances_)

# prediction of best price (Random Forest Classifier)
pred_RF = model_RF.predict(xtest)
print('prediction RF (y prediction): ', pred_RF[0])
print('y real value: ', ytest.iloc[0])


# ====================================================   KNN (K-NEAREST NEIGHBORS) ==============================================================================================================================================

from sklearn.neighbors import KNeighborsClassifier

# menentukan jumlah neighbors
# 1. sqrt(total jumlah data) = sqrt(150) = 12,2 = 13 / 11
# 2. ambil yg ganjil +1 / -1

def nNeighbors():
    x = round(len(df) ** 0.5)
    if x % 2 == 0:
        return x + 1
    else:
        return x

model_knn = KNeighborsClassifier(n_neighbors= nNeighbors())
model_knn.fit(xtrain,ytrain)

# accuracy
acc_knn = model_knn.score(xtest,ytest)
print('Accuracy KNN: ', acc_knn * 100, '%')                         # Accuracy KNN: 75 %
# print(model_RF.feature_importances_)

# prediction of best price (K-NEAREST NEIGHBORS)
pred_knn = model_knn.predict(xtest)
print('prediction KNN (y prediction): ', pred_knn[0])
print('y real value: ', ytest.iloc[0])

# # ====================================================== IMPORT JOBLIB ========================================================================================================================================================

# since the best one is KNN model, the imported model is KNN model

import joblib as jb
jb.dump(model_knn,'model_ml')





