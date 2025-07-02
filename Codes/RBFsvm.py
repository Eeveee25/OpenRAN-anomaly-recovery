# -*- coding: utf-8 -*-
"""
Created on Sat May 16 16:44:52 2020

@author: admin
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
start=time.time()
dataset=pd.read_excel('site 2.xlsx')
X= dataset.iloc[:, [0,1,2]].values
m= dataset.iloc[:, 3].values
y=np.zeros(len(m))
for j in range(len(m)):
    if m[j]=="Normal":
        y[j]=0
    elif m[j]=="medium degradation":
        y[j]=1
    else:
        y[j]=2

#split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fit svm with initial c and gamma
from sklearn.svm import SVC
SVM= SVC(kernel = 'rbf', random_state = 0 , gamma='auto')

#Gridsearch
from sklearn.model_selection import GridSearchCV  
C_range = 10. ** np.arange(-3, 5)
gamma_range = 10. ** np.arange(-3,3)
parameters = dict(gamma=gamma_range, C=C_range)

clf = GridSearchCV(estimator=SVM, param_grid=parameters ,cv=10 ,scoring='accuracy')
clf=clf.fit(X_train, y_train)
predict=clf.predict(X_test)
best_acc=clf.best_score_ 
best_C=clf.best_params_['C']
best_gamma=clf.best_params_['gamma']
  
#new c and gamma
SVM_NEW= SVC(kernel = 'rbf', random_state = 0 , gamma=best_gamma, C=best_C, decision_function_shape='ovo')
SVM_NEW.fit(X_train, y_train)
start1=time.time()
predictions_new = SVM_NEW.predict(X_test)
end1=time.time()
end=time.time()
runningcode=end-start
predicttime=end1-start1
#visualization
def visualization(X_set,y_set):
    
   
    colors = {0:'r', 1:'g',2:'b'}
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(0,np.size(X_set,0)):
        ax.scatter(X_set[i][0], X_set[i][1],X_set[i][2], c=colors[y_set[i]])

    ax.set_xlabel('call drop rate')
    ax.set_ylabel('call setup success rate')
    ax.set_zlabel('OFR_Inter X2')

  
        
# test labeled
visualization(X_test,y_test)
plt.title('Kernel SVM (Original set)') 
plt.show() 
#predictions
visualization(X_test,predictions_new)
plt.title('Kernel SVM (Prediction set)') 
plt.show() 


#Evaluation
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,predictions_new)
from sklearn.model_selection import cross_val_score
accuracy=cross_val_score(estimator=SVM_NEW ,X=X_train ,y=y_train ,cv=10)
acc_mean=accuracy.mean()
acc_std=accuracy.std()
