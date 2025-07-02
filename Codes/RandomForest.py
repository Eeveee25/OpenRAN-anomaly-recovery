import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
start = datetime.now()

#from pprint import pprint

## Importing the dataset

df = pd.read_csv("site 2.csv")
df = df.rename(columns={"OFR_Inter X2 based HO Att E-UTRAN HO Attempts, inter eNB X2 based": "HO_Attempts"})
df = df.rename(columns={"Call Drop Rate": "Call_Drop_Rate"})
df = df.rename(columns={"incoming HO succ rate": "incoming_HO_succ_rate"})
df = df.rename(columns={"Cell Availability": "Cell_Availability"})
df = df.rename(columns={"ERAB AbnormRel": "ERAB_AbnormRel"})
df = df.rename(columns={"E-UTRAN Average SINR per Cell for PUSCH": "SINR"})
df = df.rename(columns={"E-UTRAN_Average CQI": "E-UTRAN_Average_CQI"})

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
df_train, df_test= train_test_split(df, test_size = 0.25, random_state = 0)
# check purity
data = df_train.values

def check_purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False
 ####  check_purity  (df_train[df_train.petal_width < 0.8].values)

 # classification of pure data (one class )
 
def classify_data(data):  
     label_column = data[:, -1]
##l satr ali t7t ha return the unique classes ali homa l 3 w hatrg3 3dd kol class fehom
     unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
# hena hyshof akbr 3dd  fi anhon class w ya5od l argument bta3o 
     index = counts_unique_classes.argmax()
    #kda 5las 3rft a3ml classification 3la 7sb akber 3dd 
     classification = unique_classes[index] 
     
     return classification

#classify_data(df_train[df_train.petal_width < 0.8].values)

def get_potential_splits(data,rondom_subspace):
    
    potential_splits = {}
    _, n_columns = data.shape
    coiumn_indicies=list(range(n_columns-1))
    
    if rondom_subspace and rondom_subspace <= len(coiumn_indicies):
        coiumn_indicies=random.sample(population =coiumn_indicies, k=rondom_subspace)
        
    for column_index in coiumn_indicies:        # excluding the last column which is the label
        values = data[:, column_index]
        unique_values = np.unique(values)

        potential_splits[column_index]= unique_values
    
    return potential_splits

#potential_splits=get_potential_splits(df_train.values)

sns.lmplot(data=df_train, x= "HO_Attempts", y= "Call_Drop_Rate", hue="label", fit_reg=False , size=4, aspect=1.5)
#plt.vlines(x=potential_splits[3], ymin=1, ymax=7)


def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]
    data_below = data[split_column_values <= split_value]
    data_above = data[split_column_values >  split_value]
    
    return data_below, data_above

#split_column= 3
#split_value= 0.8
#data_below , data_above= split_data( data, split_column , split_value)
 
 
def calculate_entropy(data):
    
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)
# hena 7asbt l prob 3dd kol class 3la l total
    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
    return entropy    

#calculate_entropy(data_above)

def calculate_overall_entropy(data_below, data_above):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy =  (p_data_below * calculate_entropy(data_below) 
                      + p_data_above * calculate_entropy(data_above))
    
    return overall_entropy

def determine_best_split(data, potential_splits):
    
    overall_entropy = 999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)
            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value


sub_tree = {"question": ["yes_answer", "no_answer"]}


def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5,rondom_subspace = None):
    
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS
        COLUMN_HEADERS = df.columns
        data = df.values
    else:
        data = df           
    
    
    # base cases
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        
        return classification

    
    # recursive part
    else:    
        counter += 1

        # helper functions 
        potential_splits = get_potential_splits(data,rondom_subspace)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)
        
        # instantiate sub-tree
        feature_name = COLUMN_HEADERS[split_column]
        question = "{} <= {}".format(feature_name, split_value)
        sub_tree = {question: []}
        
        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth,rondom_subspace)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth,rondom_subspace)
        
        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base cases).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree
    
#tree = decision_tree_algorithm(df_train, max_depth=3)
#print(tree)



def predict_example(example, tree):
    question = list(tree.keys())[0]
    feature_name,comparison_operator, value = question.split(" ")

    # ask question
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
    # feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return predict_example(example, residual_tree)
    
#example =df_test.iloc[12]  
#predict_example(example, tree)
  
def decision_tree_predictions(test_df, tree):
    predictions =test_df.apply(predict_example , args=(tree,) ,axis=1)
    return predictions

#def calculate_accuracy(df, tree):

  #  df["classification"] = df.apply(predict_example, axis=1, args=(tree,))
   # df["classification_correct"] = df["classification"] == df["label"]
    
   # accuracy = df["classification_correct"].mean()
    
    #return accuracy



def bootstrapping(df_train, n_bootstrap):
    
    #hena ba2sm l data l different sets of data (random) hy2smha 3la 3dd l n_bootstrap
    
    bootstrap_indices = np.random.randint(low=0, high=len(df_train), size=n_bootstrap)
    df_bootstrapped = df_train.iloc[bootstrap_indices]
    
    return df_bootstrapped

#bootstrapping(df_train, n_bootstrap=10)

def random_forest_algorithm(df_train, n_trees, n_bootstrap, n_features, dt_max_depth):
    forest = []
    for i in range(n_trees):
        df_bootstrapped = bootstrapping(df_train, n_bootstrap)
        tree = decision_tree_algorithm(df_bootstrapped, max_depth=dt_max_depth,rondom_subspace = n_features)
        forest.append(tree)
    
    return forest

forest = random_forest_algorithm(df_train, n_trees=4, n_bootstrap=800, n_features=3, dt_max_depth=3)

def random_forest_predictions(test_df, forest):
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = decision_tree_predictions(test_df, tree=forest[i])
        df_predictions[column_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    random_forest_predictions = df_predictions.mode(axis=1)[0]
    
    return random_forest_predictions

predictions = random_forest_predictions(df_test, forest)

def calculate_accuracy(predictions, Label):

    predictions_correct = predictions == Label 
    accuracy = predictions_correct.mean()
    
    return accuracy
d=pd.concat([df_test,predictions],axis=1) 
d=d.rename(columns={"0": "predictions"}) 
accuracy = calculate_accuracy(predictions, df_test.label)
print('accuracy: ',accuracy)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(df_test['label'], predictions)

end = datetime.now()
time_taken = end - start
print('Time: ',time_taken)

for i in range (len(df_test)):
    if(df_test["label"].iloc[i]=='Normal'):
       df_test["label"].iloc[i]=0
    elif(df_test["label"].iloc[i]=='medium degradation'):
         df_test["label"].iloc[i]=1
    else: 
        df_test["label"].iloc[i]=2
        
for i in range (len(predictions)):
    if(predictions.iloc[i]=='Normal'):
       predictions.iloc[i]=0
    elif(predictions.iloc[i]=='medium degradation'):
        predictions.iloc[i]=1
    else: 
        predictions.iloc[i]=2        

x_test=df_test.drop(columns={"label"})
y_test=df_test['label']

colors = {0:'r', 1:'b',2:'g'}
    
fig = plt.figure()
ax = plt.axes(projection='3d')
    
for i in range(0,np.size(x_test,0)):    
     ax.scatter3D(x_test.iloc[i][0], x_test.iloc[i][1],x_test.iloc[i][2],color=colors[y_test.iloc[i]] , cmap='winter')
   
ax.set_title('degradation')   
plt.xlabel('Call_Drop_Rate')
plt.ylabel('LTE_call_setup_success_rate')
ax.set_zlabel("HO_Attempts")   
plt.show()



