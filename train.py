from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import category_encoders as ce
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import *
from models import *

df = pd.read_csv(
    "/home2/pratik_2211ai19/TechGIG/Doceree-HCP_Train.csv", encoding="latin-1"
)
new_df = data_preprocessing(df)
train, test = train_test_split(new_df)
train_x, train_y, test_x, test_y = x_y_split(train, test)
train_x_encoded, test_x_encoded = target_encoding(train, test)

dt = DecisionTreeClassifier()
accuracy_score_test, accuracy_score_train = accuracy(
    dt, train_x_encoded, test_x_encoded, train_y, test_y
)
print(
    f"(Test Accuracy : {accuracy_score_test: 4f} , Train accuracy : {accuracy_score_train: 4f})"
)

f1_score_test, f1_score_train = F1_score(
    dt, train_x_encoded, test_x_encoded, train_y, test_y
)
print(f"(Test F1 score before applying countVectorizer : {f1_score_test: 4f}, Train F1 score before applying countVectorizer : {f1_score_train:4f})")

feature_importance(dt, train_x)

final_train, final_test = Count_Vectorizer(
    df, train_x, test_x, train_x_encoded, test_x_encoded
)

dt = DecisionTreeClassifier()

accuracy_train, f1_train, accuracy_test, f1_test = classifier(dt, final_train, train_y, final_test, test_y)
print("Decision Tree Classifier")
print(f"Train accuracy is {accuracy_train} and Train F1 Score is {f1_train}",f"Test accuracy is {accuracy_test} and Test F1 Score is {f1_test}")
df = RandomForestClassifier(n_estimators=100,max_features=3, random_state=42)
print("Random Forest Classifier")
accuracy_train, f1_train, accuracy_test, f1_test = classifier(df, final_train, train_y, final_test, test_y)
print(f"Train accuracy is {accuracy_train} and Train F1 Score is {f1_train}",f"Test accuracy is {accuracy_test} and Test F1 Score is {f1_test}")
