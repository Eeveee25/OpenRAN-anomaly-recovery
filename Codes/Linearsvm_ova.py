# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 00:00:20 2020

@author: Hp
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import math 
style.use('ggplot')


dataset = pd.read_excel('site 2.xlsx')
X = dataset.iloc[:, [0,1,2]].values
m= dataset.iloc[:, 3].values

y_train_new1=[]
y_train_new2=[]
y_train_new3=[]
swap=np.zeros(1307)
swap2=np.zeros(1307)
swap3=np.zeros(1307)
y_predicted=np.zeros(436)
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
   
#one vs all               
for j in range(len(y_train)):
    if y_train[j]==0:
        swap[j]=1
    if (y_train[j]==1 or y_train[j]==2):
        swap[j]=-1
y_train_new1=swap
    #####1st class
for j in range(len(y_train)):
    if y_train[j]==1:
         swap2[j]=1
    if (y_train[j]==0 or y_train[j]==2):
        swap2[j]=-1
y_train_new2=swap2
    ##second class
for j in range(len(y_train)):
    if y_train[j]==2:
         swap3[j]=1
    if (y_train[j]==1 or y_train[j]==0):
         swap3[j]=-1
y_train_new3= swap3
    ##############third class
    
    # Function to find distance
def prependicular_distance(x1, y1, x_coef, y_coef, const):         
    dis = abs(((x_coef * x1) + (y_coef * y1) - const)) / (math.sqrt((x_coef * x_coef) + (y_coef * y_coef))) 
    return dis
       
#creat model
class S_V_M:
    
    def __init__(self, learning_rate, lambda_param, n_iters):
        self.lr = learning_rate
        self.lambda_reg = lambda_param
        self.n_iteration = n_iters
    def fit(self, X_train, y_train):
        #initial
        self.w1 = np.zeros(3)
        self.b1 = 0
        self.w2 = np.zeros(3)
        self.b2 = 0
        self.w3 = np.zeros(3)
        self.b3 = 0
        ## gradient
        for p in range(self.n_iteration):
            for idx, x_i in enumerate(X_train):
                condition = y_train_new1[idx] * (np.dot(x_i, self.w1) - self.b1) >= 1
                if condition:
                    self.w1 -= self.lr * (2 * self.lambda_reg * self.w1)
                else:
                    self.w1 -= self.lr * (2 * self.lambda_reg * self.w1 - np.dot(x_i, y_train_new1[idx]))
                    self.b1 -= self.lr * y_train_new1[idx]
        for p in range(self.n_iteration):
            for idx, x_i in enumerate(X_train):
                condition = y_train_new2[idx] * (np.dot(x_i, self.w2) - self.b2) >= 1
                if condition:
                    self.w2 -= self.lr * (2 * self.lambda_reg * self.w2)
                else:
                    self.w2 -= self.lr * (2 * self.lambda_reg * self.w2 - np.dot(x_i, y_train_new2[idx]))
                    self.b2 -= self.lr * y_train_new2[idx]
        for p in range(self.n_iteration):
            for idx, x_i in enumerate(X_train):
                condition = y_train_new3[idx] * (np.dot(x_i, self.w3) - self.b3) >= 1
                if condition:
                    self.w3 -= self.lr * (2 * self.lambda_reg * self.w3)
                else:
                    self.w3 -= self.lr * (2 * self.lambda_reg * self.w3 - np.dot(x_i, y_train_new3[idx]))
                    self.b3 -= self.lr * y_train_new3[idx]
                    

                                        
     #predict 
                 
    def predict(self,X_pre):
        self.x=X_pre
        for i in range (len(X_pre)) :
            d1=prependicular_distance(X_pre[i,0], X_pre[i,1], self.w1[0], self.w1[1], self.b1)
            d2=prependicular_distance(X_pre[i,0], X_pre[i,1], self.w2[0], self.w2[1], self.b2)
            d3=prependicular_distance(X_pre[i,0], X_pre[i,1], self.w3[0], self.w3[1], self.b3)
            if np.sign(np.dot(self.x[i],self.w1)-self.b1)>0 and np.sign(np.dot(self.x[i],self.w2)-self.b2)>0: 
                if d1<d2:
                    y_predicted[i]= 0              
                else:
                    y_predicted[i]= 1
                    
            elif np.sign(np.dot(self.x[i],self.w1)-self.b1)>0 and np.sign(np.dot(self.x[i],self.w3)-self.b3)>0: 
                if d1<d3:
                    y_predicted[i]= 0              
                else:
                    y_predicted[i]= 2
            elif np.sign(np.dot(self.x[i],self.w2)-self.b2)>0 and np.sign(np.dot(self.x[i],self.w3)-self.b3)>0: 
                if d2<d3:
                    y_predicted[i]= 1              
                else:
                    y_predicted[i]= 2
            elif np.sign(np.dot(self.x[i],self.w1)-self.b1)>0:    
                 y_predicted[i]=0
            elif np.sign(np.dot(self.x[i],self.w2)-self.b2)>0:
                 y_predicted[i]=1
            elif np.sign(np.dot(self.x[i],self.w3)-self.b3)>0:
                 y_predicted[i]=2
            else:
                distances=[d1,d2,d3]
                y_predicted[i]= (distances.index(min( distances)))
        return y_predicted
                
               
# Function to find distance 
def visualize_svm(a,m):
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.scatter(a[:,0], a[:,1] ,marker='o',c=m)
     
    x0_1 = np.amin(X_train[:,0])
    x0_2 = np.amax(X_train[:,0])
    
    x1_1 = get_hyperplane_value(x0_1, svm.w1, svm.b1, 0)
    x1_2 = get_hyperplane_value(x0_2, svm.w1, svm.b1, 0)
    
    x1_1_m = get_hyperplane_value(x0_1, svm.w2, svm.b2, 0)
    x1_2_m = get_hyperplane_value(x0_2, svm.w2, svm.b2, 0)
    
    x1_1_p = get_hyperplane_value(x0_1, svm.w3, svm.b3, 0)
    x1_2_p = get_hyperplane_value(x0_2, svm.w3, svm.b3, 0)
    
    ax.plot([x0_1, x0_2],[x1_1, x1_2], 'k',[x0_1, x0_2],[x1_1_m, x1_2_m], 'y',[x0_1, x0_2],[x1_1_p, x1_2_p], 'r')
    
    x1_min = np.amin(X_train[:,1])
    x1_max = np.amax(X_train[:,1])
    ax.set_ylim([x1_min-3,x1_max+3])
    plt.show()

  
   
svm = S_V_M(0.01,0.0001,100)
svm.fit(X_train, y_train)
predictions=svm.predict(X_test) 
visualize_svm(X_train,y_train)
visualize_svm(X_test,predictions)
#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
from sklearn import metrics
acc=metrics.accuracy_score(y_test, predictions)
