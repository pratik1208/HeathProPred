from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import category_encoders as ce
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def train_test_split(df):
    test = df.sample(20000, random_state=42)
    train = df[~df.index.isin(test.index)]
    return train, test


def x_y_split(train, test):
    train_x = train.iloc[:, :-1]
    train_y = train.iloc[:, -1]
    test_x = test.iloc[:, :-1]
    test_y = test.iloc[:, -1]
    return train_x, train_y, test_x, test_y


def data_preprocessing(df):
    df["KEYWORDS"] = df["KEYWORDS"].str.lower()
    new_df = df.drop(
        [
            "TAXONOMY",
            "KEYWORDS",
            "CHANNELTYPE",
            "PLATFORMTYPE",
            "USERPLATFORMUID",
            "BIDREQUESTIP",
            "ID",
            "PLATFORM_ID",
            "DEVICETYPE",
        ],
        axis=1,
    )
    new_df = new_df.fillna(0)
    return new_df




def accuracy(model, train_x, test_x, train_y, test_y):
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    y_pred_train = model.predict(train_x)
    return (accuracy_score(y_pred, test_y), (accuracy_score(y_pred_train, train_y)))


def F1_score(model, train_x, test_x, train_y, test_y):
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    y_pred_train = model.predict(train_x)
    return (f1_score(y_pred, test_y), (f1_score(y_pred_train, train_y)))


def feature_importance(model, data):
    importance = model.feature_importances_
    feature_names = data.columns
    import matplotlib.pyplot as plt

    plt.barh(range(len(importance)), importance, align="center")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.yticks(range(len(importance)), feature_names)
    for index, value in enumerate(importance):
        plt.text(value, index, f"{value:.4f}")
    plt.show()


def total_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def encode(data_frame,encoder):
    df_copy = data_frame.copy()
    for col in data_frame.columns:
        if col in encoder:
            df_copy[col] = df_copy[col].apply(
                lambda x: encoder[col][x]
                if x and (x is not np.nan) and (x in encoder[col]) else 0
            )
    return df_copy