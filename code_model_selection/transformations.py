# Upsample the minority class (brakes applied)
import pandas as pd
from sklearn.utils import resample

def upsample_data(input_df,col="Brake",majority_value=0):
    '''
    Creates a balanced data set from the dataframe provided to it by upsampling the
    minority class, using col as the column_name of classes to be balanced
    '''    

    #Split by row based on the data class
    df_majority = input_df[input_df[col]==majority_value]
    df_minority = input_df[input_df[col]!=majority_value]
    
    df_minority_upsampled = resample(df_minority,
                                  replace=True,
                                  n_samples=df_majority.shape[0],
                                  random_state=444)
    
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    #print("Before upsampling:\n",df.Brake.value_counts())
    #print("After upsampling:\n",df_upsampled.Brake.value_counts())
    #print(df_upsampled.describe())
    
    return df_upsampled
