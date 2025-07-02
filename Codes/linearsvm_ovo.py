# -*- coding: utf-8 -*-
"""
Created on Sun May 17 04:20:24 2020

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 15 23:49:16 2020

@author: Hp
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import math

style.use('ggplot')
dataset=pd.read_excel('site 2.xlsx')
X= dataset.iloc[:, [0,1,2]].values
m=dataset.iloc[:, 3]
y=np.zeros(len(m))
for i in range (len(m)):
    if(m[i]=='Normal'):
        y[i]=0
    elif(m[i]=='medium degradation'):
         y[i]=1
    else: 
        y[i]=2
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

y_train_new11=[]
y_train_new22=[]
y_train_new33=[]
y_train_new1=[]
y_train_new2=[]
y_train_new3=[]
x_train_new11=[]
x_train_new22=[]
x_train_new33=[]
x_train_new1=[]
x_train_new2=[]
x_train_new3=[]
y_predicted=np.zeros(len(y_test))
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# one vs one               
for j in range(len(y_train)):
    if y_train[j]==0:
        y_train_new1.append(1)
        y_train_new11.append(1)
        x_train_new1.append(X_train[j,:])
        x_train_new11.append(X_train[j,:])
    elif y_train[j]==1:
        y_train_new1.append(-1)
        x_train_new1.append(X_train[j,:])
    else:
        y_train_new11.append(-1)
        x_train_new11.append(X_train[j,:])
        

for j in range(len(y_train)):
    if y_train[j]==1:
        y_train_new2.append(1)
        y_train_new22.append(1)
        x_train_new2.append(X_train[j,:])
        x_train_new22.append(X_train[j,:])
    elif y_train[j]==0:
        y_train_new2.append(-1)
        x_train_new2.append(X_train[j,:])
    else:
        y_train_new22.append(-1)
        x_train_new22.append(X_train[j,:])

for j in range(len(y_train)):
    if y_train[j]==2:
        y_train_new3.append(1)
        y_train_new33.append(1) 
        x_train_new3.append(X_train[j,:])
        x_train_new33.append(X_train[j,:])
    elif y_train[j]==1:
        y_train_new3.append(-1)
        x_train_new3.append(X_train[j,:])
    else:
        y_train_new33.append(-1)
        x_train_new33.append(X_train[j,:])


y_train_new11=np.array(y_train_new11)
y_train_new22=np.array(y_train_new22)
y_train_new33=np.array(y_train_new33)
y_train_new1=np.array(y_train_new1)
y_train_new2=np.array(y_train_new2)
y_train_new3=np.array(y_train_new3)
x_train_new11=np.array(x_train_new11)
x_train_new22=np.array(x_train_new22)
x_train_new33=np.array(x_train_new33)
x_train_new1=np.array(x_train_new1)
x_train_new2=np.array(x_train_new2)
x_train_new3=np.array(x_train_new3)


    # Function to find distance
def prependicular_distance(x1, y1,z1, x_coef, y_coef,z_coef, const):         
    dis = abs(((x_coef * x1) + (y_coef * y1) + (z_coef * z1) - const)) / (math.sqrt((x_coef * x_coef) + (y_coef * y_coef) + (z_coef * z_coef))) 
    return dis
    
class S_V_M:
    
    def __init__(self, learning_rate, lambda_param, n_iters):
        self.lr = learning_rate
        self.lambda_reg = lambda_param
        self.n_iteration = n_iters
        
    def fit(self):
        #initial
        self.w1 = np.zeros(3)
        self.b1 = 0
        self.w2 = np.zeros(3)
        self.b2 = 0
        self.w3 = np.zeros(3)
        self.b3 = 0
        self.w11 = np.zeros(3)
        self.b11 = 0
        self.w22 = np.zeros(3)
        self.b22 = 0
        self.w33 = np.zeros(3)
        self.b33 = 0
        ## gradient descent
        for p in range(self.n_iteration):
            for idx, x_i in enumerate(x_train_new1):
                condition1 = y_train_new1[idx] * (np.dot(x_i, self.w1) - self.b1) >= 1
                if condition1:
                    self.w1 -= self.lr * (2 * self.lambda_reg * self.w1)
                else:
                    self.w1 -= self.lr * (2 * self.lambda_reg * self.w1 - np.dot(x_i, y_train_new1[idx]))
                    self.b1 -= self.lr * y_train_new1[idx]
                   
            for idx, x_i in enumerate(x_train_new11):        
                condition11 = y_train_new11[idx] * (np.dot(x_i, self.w11) - self.b11) >= 1
                if condition11:
                     self.w11 -= self.lr * (2 * self.lambda_reg * self.w11)           
                else:
                    self.w11 -= self.lr * (2 * self.lambda_reg * self.w11 - np.dot(x_i, y_train_new11[idx]))
                    self.b11 -= self.lr * y_train_new11[idx]
            
        
            for idx, x_i in enumerate(x_train_new2):
                condition2 = y_train_new2[idx] * (np.dot(x_i, self.w2) - self.b2) >= 1
                if condition2:
                    self.w2 -= self.lr * (2 * self.lambda_reg * self.w2)
                else:
                    self.w2 -= self.lr * (2 * self.lambda_reg * self.w2 - np.dot(x_i, y_train_new2[idx]))
                    self.b2 -= self.lr * y_train_new2[idx]
                    
                    
            for idx, x_i in enumerate(x_train_new22):   
                condition22 = y_train_new22[idx] * (np.dot(x_i, self.w22) - self.b22) >= 1
                if condition22:
                    self.w22 -= self.lr * (2 * self.lambda_reg * self.w22)
                else:
                    self.w22 -= self.lr * (2 * self.lambda_reg * self.w22 - np.dot(x_i, y_train_new22[idx]))
                    self.b22 -= self.lr * y_train_new22[idx]
        
            for idx, x_i in enumerate(x_train_new3):
                condition3 = y_train_new3[idx] * (np.dot(x_i, self.w3) - self.b3) >= 1
                if condition3:
                    self.w3 -= self.lr * (2 * self.lambda_reg * self.w3)
                else:
                    self.w3 -= self.lr * (2 * self.lambda_reg * self.w3 - np.dot(x_i, y_train_new3[idx]))
                    self.b3 -= self.lr * y_train_new3[idx]
                    
                   
            for idx, x_i in enumerate(x_train_new33):       
                condition33 = y_train_new33[idx] * (np.dot(x_i, self.w33) - self.b33) >= 1
                if condition33:
                    self.w33 -= self.lr * (2 * self.lambda_reg * self.w33)
                else:
                    self.w33 -= self.lr * (2 * self.lambda_reg * self.w33 - np.dot(x_i, y_train_new33[idx]))
                    self.b33 -= self.lr * y_train_new33[idx]

                                               
     #predict               
    def predict(self,X_pre):
        self.x=X_pre
        for i in range (len(X_pre)) :
            cond1=np.sign(np.dot(self.x[i],self.w1)-self.b1)>0 and np.sign(np.dot(self.x[i],self.w11)-self.b11)>0
            cond2=np.sign(np.dot(self.x[i],self.w2)-self.b2)>0 and np.sign(np.dot(self.x[i],self.w22)-self.b22)>0
            cond3=np.sign(np.dot(self.x[i],self.w3)-self.b3)>0 and np.sign(np.dot(self.x[i],self.w33)-self.b33)>0
            d1=prependicular_distance(X_pre[i,0], X_pre[i,1],X_pre[i,2], self.w1[0], self.w1[1], self.w1[2], self.b1)
            d2=prependicular_distance(X_pre[i,0], X_pre[i,1],X_pre[i,2], self.w2[0], self.w2[1], self.w2[2], self.b2)
            d3=prependicular_distance(X_pre[i,0], X_pre[i,1],X_pre[i,2], self.w3[0], self.w3[1], self.w3[2], self.b3)
            d11=prependicular_distance(X_pre[i,0], X_pre[i,1],X_pre[i,2], self.w11[0], self.w11[1], self.w11[2], self.b11)
            d22=prependicular_distance(X_pre[i,0], X_pre[i,1],X_pre[i,2], self.w22[0], self.w22[1], self.w22[2], self.b22)
            d33=prependicular_distance(X_pre[i,0], X_pre[i,1],X_pre[i,2], self.w33[0], self.w33[1], self.w33[2], self.b33)
            class1=min(d1,d11)
            class2=min(d2,d22)
            class3=min(d3,d33)
            if cond1 and cond2:  
                 
                 if class1<class2  :
                    y_predicted[i]= 0              
                 else:
                    y_predicted[i]= 1
            elif cond1 and cond3:
                 
                 if class1<class3  :
                    y_predicted[i]= 0              
                 else:
                    y_predicted[i]= 2
            elif cond2 and cond3:
                 
                 if class2<class3  :
                    y_predicted[i]= 1             
                 else:
                    y_predicted[i]= 2
            elif cond1:
                 y_predicted[i]=0
            elif cond2:
                 y_predicted[i]=1
            elif cond3:
                 y_predicted[i]=2
            else:    
                
                classes=[(d1+d11)/2,(d2+d22)/2,(d3+d33)/2]
               
                y_predicted[i]= (classes.index(min(classes)))
                
               
        return y_predicted
     
#visualization
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


svm = S_V_M(0.01,0.001,100)
svm.fit()
predictions=svm.predict(X_test)
visualize_svm(X_train,y_train)
visualize_svm(X_test,predictions)
#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
from sklearn import metrics
acc=metrics.accuracy_score(y_test, predictions)

