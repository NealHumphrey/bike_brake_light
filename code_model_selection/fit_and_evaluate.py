from sklearn import metrics
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from transformations import upsample_data
import numpy as np
import pandas as pd


def fit_and_evaluate(
        model,
        label,
        input_data,
        target_column = "Brake",
        # Scenario parameters
        upsample = True,
        ):
    """
    Performs K-fold cross validation to create our evaluation scores, and then retrains the model
    on the entire data set. 
    
    df_data: a dataframe of the data to be modeled, with 'y' as the last column
    model: the sklearn model class that we want to create a new instance of
    label: string printed above the output; not stored in any way
    """
    
    #setup output variables
    scores={'precision':[],'recall':[],'f1':[]}
    cm_list=[] #confusion_matrix
    
    
    #Perform cross validation training
    for train, test in KFold(input_data.shape[0], n_folds=12,shuffle=True):

        train_data = input_data.iloc[train]
        test_data = input_data.iloc[test]
        
        if upsample == True:
            train_data = upsample_data(train_data,col=target_column,majority_value=0)

        y_train = train_data.iloc[:,-1] # get just target_column
        X_train = train_data.drop(target_column,1) # everything except target
        
        y_test = test_data.iloc[:,-1] 
        X_test = test_data.drop(target_column,1)

        model.fit(X_train, y_train)
        
        expected  = y_test
        predicted = model.predict(X_test)
        
        # Append our scores to the tracker
        scores['precision'].append(metrics.precision_score(expected, predicted, average="weighted"))
        scores['recall'].append(metrics.recall_score(expected, predicted, average="weighted"))
        scores['f1'].append(metrics.f1_score(expected, predicted, average="weighted"))
        
        cm_list.append(confusion_matrix(expected,predicted))
    
    # Aggregate Confusion Matrix
    cm = np.array([[0,0],[0,0]])
    for c in cm_list:
        cm = np.add(cm,c)
    
    # Retrain the model on the whole data set
    full_train_data = input_data
    if upsample == True:
        full_train_data = upsample_data(full_train_data,col=target_column,majority_value=0)

    full_y = full_train_data.iloc[:,-1]
    full_X = full_train_data.drop(target_column,1)

    model.fit(full_X, full_y)
    
    print(label)
    print('-----------------')
    print(pd.DataFrame(scores).mean())
    print('--Confusion Matrix (k-fold aggregate)--')
    print(cm)
    
    return model,scores,cm