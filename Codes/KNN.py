# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:43:41 2020

@author: Emad
"""
from datetime import datetime

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Importing the dataset
#from sklearn.datasets import load_iris
#iris=load_iris()
#X=iris.data[:,:2]
#y=iris.target
dataset = pd.read_csv('site 2.csv')
X = dataset.iloc[:,[0,1,2]].values
yy = dataset.iloc[:, 3]
#     
# Splitting the dataset into the Training set and Test set
nn = np.size(X,0)
m = nn*0.25
mm = int(m)
#X_train1 = np.copy(X[mm:,:])
#X_test1 = np.copy(X[:mm,:])
#y=np.reshape(y,(nn,1))
#y_train1 = np.copy(y[mm:,:])
#y_test1 = np.copy(y[:mm,:])

#categorical_data/// 0_normal//2_medium_degradation//1_critical_degradation
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(yy)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train1)
X_test = sc.transform(X_test1)

n = np.size(X_train,0)
Y_train=np.reshape(y_train1,(n,1))
train = np.append(X_train,Y_train,axis=1)
     
class Knn:
    def __init__(self, k):
        self.__k = k
    def get_neighbors(self, training_set, test_instance):
        no_of_training_instances = np.size(training_set,0) #0 to count the rows
        no_of_training_columns = np.size(training_set,1) #1 to count the attributes
        actual_class_column = None
        actual_class_column = no_of_training_columns-1
        actual_class_and_distance = np.zeros((no_of_training_instances, 2))
        neighbors = None
        for row in range(0, no_of_training_instances):
            actual_class_and_distance[row,0] = training_set[row,actual_class_column]
            temp_training_instance =np.reshape( np.copy(training_set[row,[0,1,2]]),(1,3))
            temp_test_instance = np.copy(test_instance)
            z=np.subtract(temp_test_instance,temp_training_instance)
            distance = np.linalg.norm(z) # calculating Euclidean distance
            actual_class_and_distance[row,1] = distance
        actual_class_and_distance = actual_class_and_distance[actual_class_and_distance[:,1].argsort()]
        k = self.__k
        neighbors = actual_class_and_distance[:k,:]
        return neighbors
    def make_prediction(self, neighbors):
        prediction = None
        neighborsint = neighbors.astype(int)
        prediction = np.bincount(neighborsint).argmax()
        return prediction
    def get_accuracy(self, actual_class_array, predicted_class_array):
        accuracy = None
        counter = None
        actual_class_array_size = np.size(actual_class_array,0)
        counter = 0
        for row in range(0,actual_class_array_size):
            if actual_class_array[row] == predicted_class_array[row]:
                counter += 1
        accuracy = counter / (actual_class_array_size) 
        return accuracy*100
#class ScatterPlot(panels.Plot):
#    name = "Scatter"
#    def plot(self, inputs):
#        p = ggplot(meat, aes(x='date', y=inputs.yvar))
#        if inputs.smoother:
#            p = p + stat_smooth(color="blue")
#        p = p + geom_point() + ggtitle(inputs.title)
#        return p
#    
knn1 = Knn(16)
no_of_test_instances= np.size(X_test,0)
predicted_class_values=np.zeros((no_of_test_instances, 1))
#predicted_class_values=np.zeros(no_of_test_instances)
start = datetime.now()
for row in range(0, no_of_test_instances):
    this_instance = X_test[row,:]
    this_instance = np.reshape(this_instance,(1,3))
    neighbor_array = knn1.get_neighbors(train,this_instance)
    neighbors_arr = neighbor_array[:,0]
    prediction = knn1.make_prediction(neighbors_arr)
    predicted_class_values[row] = prediction
end = datetime.now()
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test1, predicted_class_values)
 
accu= Knn.get_accuracy (knn1,y_test1, predicted_class_values)
colors = {0:'r', 1:'g',2:'b'}

fig = plt.figure()
ax = plt.axes(projection='3d')

for i in range(0,np.size(X_test1,0)):
    ax.scatter3D(X_test1[i][0], X_test1[i][1],X_test1[i][2],color=colors[predicted_class_values[i][0]])
ax.set_title('degradation')

plt.xlabel('Call Drop Rate')
plt.ylabel('LTE_call_setup_success_rate')
#plt.zlabel('OFR_Inter X2')

time_taken = end - start
print('Time: ',time_taken)