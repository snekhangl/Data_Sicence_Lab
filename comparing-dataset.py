import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# import the library
pima_dia = pd.read_csv('pima indians diabetes.csv')
uci_dia = pd.read_csv('UCI-diabetes.csv')
df1=pima_dia[['BMI','Age']]
df1["BMI"] = np.where(df1["BMI"] > 30,1,0)
df2=uci_dia[['Obesity','Age']]
df2["Obesity"] = np.where(df2["Obesity"] == 'Yes', 1,0)
uci_dia["class"] = np.where(uci_dia["class"] == 'Positive', 1,0)
X = df1.values         #independent variable array
Y = pima_dia.iloc[:,8].values #dependent variable vector 
X1 = df2.values         #independent variable array
Y1 = uci_dia.iloc[:,16].values #dependent variable vector 
#splitting
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20,random_state=0)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, Y1, test_size=0.20,random_state=0)
#fitting the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
train_acc = model.score(X_train,y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred)
out = model.predict([[1,45]])
ou = model.predict([[0,25]])
model1 = LogisticRegression()
model1.fit(X1_train,y1_train)
train_acc1 = model1.score(X1_train,y1_train)
y1_pred = model1.predict(X1_test)
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y1_test, y1_pred)
out1 = model1.predict([[1,45]])
ou1 = model1.predict([[0,25]])
print('\n\n')
print('                         Pima indians dataset           UCI diabetes dataset\n')
print('Accuracy  Test set       : ', train_acc*100, '         ', train_acc1*100,'\n')
print('Accur-Logi regression    : ', model.score(X_test, y_test), '        ', model1.score(X1_test, y1_test),'\n')
print('Confusion matrix         : ', cm1[0][0],'    ',cm1[0][1],'\t\t\t\t\t', cm2[0][0],'    ',cm2[0][1],'\n\t\t\t\t\t\t   ', cm1[1][0],'    ',cm1[1][1],'\t\t\t\t\t', cm2[1][0],'    ',cm2[1][1],'\n' )
print('outcome obes 1 & age 45  :','', out, '                        ', out1,'\n\n') 
print('outcome obes 0 & age 25  :','', ou, '                        ', ou1,'\n\n') 
