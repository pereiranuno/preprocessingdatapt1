# coding: utf-8
from typing import Tuple

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score


def split_dataset(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into two dataframes, one for training and one for testing.
    
    Args:
        df: dataframe with the dataset
        target: name of the column with the target variable
    
    Return: 
        Tuple with (train set dataframe, test set dataframe)
    """
    # for input features we keep all columns except the target column 
    input_columns = [col for col in df.columns if col!=target]
    X = df[input_columns]
    y = df[target]
    # splits the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2018)
    # joins the X and y into a single dataframe for both train and test
    train_df = X_train
    train_df[target] = y_train
    test_df = X_test
    test_df[target] = y_test
    return train_df, test_df

def score_dataset(X_train: pd.DataFrame, X_test: pd.Series, y_train: pd.DataFrame, y_test: pd.Series) -> float:
    model = linear_model.LogisticRegression(solver='liblinear')

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return accuracy_score(y_test, pred)

def score_approach(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str) -> float:
    """
    Trains the model with train_df and evaluates it in test_df, returning the accuracy.
    This is helper to score a given approach to deal with missing values
    
    Args:
        train_df: dataframe with the train dataset
        test_df: dataframe with the test dataset
        target: name of the column with the target variable
    
    Return: 
        test set accuracy
    """
    # for input features we keep all columns except the target column 
    input_columns = [col for col in train_df.columns if col!=target]
    X_train = train_df[input_columns]
    y_train = train_df[target]
    X_test = test_df[input_columns]
    y_test = test_df[target]
    # train the model and get the accuracy in the test set
    return score_dataset(X_train, X_test, y_train, y_test)

def calculate_iv(df, feature, target, bins=5):
    df['bin'] = pd.qcut(df[feature], bins, duplicates='drop')
    grouped = df.groupby('bin', observed=True)[target].agg(['count', 'sum'])
    grouped['non_event'] = grouped['count'] - grouped['sum']
    grouped['event_rate'] = grouped['sum'] / grouped['sum'].sum()
    grouped['non_event_rate'] = grouped['non_event'] / grouped['non_event'].sum()
    grouped['woe'] = np.log(grouped['event_rate'] / grouped['non_event_rate'])
    grouped['iv'] = (grouped['event_rate'] - grouped['non_event_rate']) * grouped['woe']
    iv = grouped['iv'].sum()
    return iv