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


def target_encoding(train, test):
    train_x = train.iloc[:, :-1]
    train_y = train.iloc[:, -1]
    test_x = test.iloc[:, :-1]
    test_y = test.iloc[:, -1]
    encoder = ce.TargetEncoder(cols=train_x.columns, smoothing=0.2)
    train_x_encoded = encoder.fit_transform(train_x, train_y)
    test_x_encoded = encoder.transform(test_x, test_y)
    return train_x_encoded, test_x_encoded


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


def Count_Vectorizer(df, train_x, test_x, train_x_encoded, test_x_encoded):
    train_x = train_x.join(df.KEYWORDS, how="left").copy()
    test_x = test_x.join(df.KEYWORDS, how="left").copy()
    vectorizer = CountVectorizer(max_features=514)
    train_x_sparse_matrix = vectorizer.fit_transform(train_x.KEYWORDS)
    test_x_sparse_matrix = vectorizer.transform(test_x.KEYWORDS)
    feature_names = vectorizer.get_feature_names_out()
    df_train = pd.DataFrame(
        train_x_sparse_matrix.toarray(), columns=feature_names, index=train_x.index
    )
    df_test = pd.DataFrame(
        test_x_sparse_matrix.toarray(), columns=feature_names, index=test_x.index
    )
    final_train = train_x_encoded.join(df_train, how="left")
    final_test = test_x_encoded.join(df_test, how="left")
    return final_train, final_test


def evaluate(model, data_loader, device="cpu"):
    model = model.eval()
    with torch.inference_mode():
        preds, reals = [], []
        for features, targets in tqdm(data_loader):
            pred = model(features.to(torch.float32).to(device))
            preds.append(pred.detach().cpu())
            reals.append(targets)
        pred = torch.cat(preds) > 0.5
        real = torch.cat(reals)
        f1 = f1_score(pred, real, average="macro")
        accuracy = accuracy_score(pred, real)

    model.train()
    return f1, accuracy
