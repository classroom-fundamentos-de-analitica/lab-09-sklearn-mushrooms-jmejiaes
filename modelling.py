import pandas as pd
import numpy as np
import os
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_datasets():
    """Load train and test datasets."""
    train_dataset = pd.read_csv("train_dataset.csv")
    test_dataset = pd.read_csv("test_dataset.csv")
    
    x_train = train_dataset.drop("type", axis=1)
    y_train = train_dataset["type"]
    
    x_test = test_dataset.drop("type", axis=1)
    y_test = test_dataset["type"]


    return x_train, x_test, y_train, y_test

def encode_df(df, df1 = None):
    
    if type(df) == pd.core.series.Series:

        df = df.to_frame()
        df.columns = ['type']
        df1 = df1.to_frame()
        df1.columns = ['type']    



    #Here we extract the columns with object datatype as they are the categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    #Initialize OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)

    # Apply one-hot encoding to the categorical columns
    one_hot_encoded = encoder.fit_transform(df[categorical_columns])    

    #Create a DataFrame with the one-hot encoded columns
    #We use get_feature_names_out() to get the column names for the encoded data
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
        
    # Concatenate the one-hot encoded dataframe with the original dataframe
    df_encoded = pd.concat([df, one_hot_df], axis=1)

    # Drop the original categorical columns
    df_encoded = df_encoded.drop(categorical_columns, axis=1)

    if df1 is not None:
        one_hot_encoded1 = encoder.transform(df1[categorical_columns])    

        one_hot_df1 = pd.DataFrame(one_hot_encoded1, columns=encoder.get_feature_names_out(categorical_columns))
        
        df_encoded1 = pd.concat([df1, one_hot_df1], axis=1)

        df_encoded1 = df_encoded1.drop(categorical_columns, axis=1)

        return df_encoded, df_encoded1
    
    else:
        return df_encoded


def process_data(x_train, x_test, y_train, y_test):
    """Process datasets."""



    encoder = OneHotEncoder()

    x_train_encoded = pd.DataFrame(encoder.fit_transform(x_train[categorical_columns]))
    x_test_encoded = pd.DataFrame(encoder.transform(x_test[categorical_columns]))

    encoder2 = OneHotEncoder()

    y_train_encoded = pd.DataFrame(encoder2.fit_transform(y_train.values.reshape(-1, 1)))
    y_test_encoded = pd.DataFrame(encoder2.transform(y_test.values.reshape(-1, 1)))
                                  
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    print(x_train_encoded.shape, x_test_encoded.shape, y_train_encoded.shape, y_test_encoded.shape)

    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):
    """Train a logistic regression model."""
    model = LogisticRegression()

    
    model.fit(x_train, y_train)
    
    return model

def save_model(model):
    """Save trained model to disk."""
    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)
    
    

def run_grading():
    """Run grading script."""
    os.system("python test.py")

def pipeline_for_model(x_train, x_test, y_train, y_test):
    x_train, x_test = encode_df(x_train, x_test)

    y_train, y_test = encode_df(y_train, y_test)

    y_train = y_train['type_p']
    y_test = y_test['type_p']

    return x_train, x_test, y_train, y_test

if __name__ == "__main__":



    
    x_train, x_test, y_train, y_test =  load_datasets() 

    x_train, x_test = encode_df(x_train, x_test)

    y_train, y_test = encode_df(y_train, y_test)

    y_train = y_train['type_p']
    y_test = y_test['type_p']

    model = train_model(x_train, y_train)
    save_model(model)
    
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    
    print(f"Train accuracy: {accuracy_train:.2f}")
    print(f"Test accuracy: {accuracy_test:.2f}")
