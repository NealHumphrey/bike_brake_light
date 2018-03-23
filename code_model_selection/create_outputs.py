import pandas as pd
from sklearn.metrics import confusion_matrix

def create_comparison_file(estimator,output_name,target_column="Brake",test_data_path='data/2018-01-31.csv'):
    """
    Takes a fitted model, runs it on our unseen data set, and outputs the resulting predictions to a file.
    """
    test_df = pd.read_csv(test_data_path, sep=',', header=0)
    
    # Run through same processing steps! (pipeline)
    test_df = test_df.iloc[:,3:]
    

    
    test_y_actual = test_df.iloc[:,-1]
    test_X = test_df.drop(target_column,1)

    test_y_predicted = estimator.predict(test_X)
    
    cm = confusion_matrix(test_y_actual,test_y_predicted)
    print('--Confusion Matrix (test data)--')
    print(cm)
    
    test_comparison_df = pd.DataFrame({'actual':test_y_actual,'predicted':test_y_predicted})
    output_path = 'outputs/' + output_name
    test_comparison_df.to_csv(output_path);