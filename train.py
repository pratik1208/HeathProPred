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
from utils import *
from models import *
from torch.utils.data import Dataset, DataLoader


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

f1_score_test, f1_score_train = F1_score(
    dt, train_x_encoded, test_x_encoded, train_y, test_y
)

feature_importance(dt, train_x)

final_train, final_test = Count_Vectorizer(
    df, train_x, test_x, train_x_encoded, test_x_encoded
)

neuralnet = NeuralNetwork()

features = final_train
targets = train_y
my_dataset = MyDataset(features, targets)

batch_size = 32
shuffle = True

my_dataloader = DataLoader(dataset=my_dataset, batch_size=batch_size, shuffle=shuffle)
device = torch.device("cuda")
neuralnet = neuralnet.to(device)
epochs = 20
optimizer = torch.optim.Adam(neuralnet.parameters(), lr=0.001)
loss = nn.BCELoss()

train_epoch = []
train_accuracy = []
train_f1_score = []
for epoch in tqdm(range(epochs)):

    for batch in tqdm(my_dataloader):
        batch_features, batch_targets = batch
        pred = neuralnet(batch_features.to(torch.float32).to(device))
        l = loss(pred.squeeze(), batch_targets.to(torch.float32).to(device))
        print(f"\rloss = {l.item()}", end=" ")
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    f1, accuracy = evaluate(neuralnet, my_dataloader, device)
    train_epoch.append(epoch)
    train_accuracy.append(accuracy)
    train_f1_score.append(f1)
    print(
        f"Train Accuracy at {epoch} : {accuracy} and F1 Score : {f1}"
    )

test_features = final_test
test_target = test_y

my_dataset_test = MyDataset(test_features, test_target)

batch_size = 32
shuffle = True
my_dataloader_test = DataLoader(
    dataset=my_dataset_test, batch_size=batch_size, shuffle=shuffle
)

evaluate(neuralnet, my_dataloader_test, device)

optimizer = torch.optim.Adam(neuralnet.parameters(), lr=0.001)
loss = nn.BCELoss()
test_epoch = []
test_accuracy = []
test_f1_score = []
epochs = 20
for epoch in tqdm(range(epochs)):

    for batch in tqdm(my_dataloader_test):
        batch_features, batch_targets = batch
        pred = neuralnet(batch_features.to(torch.float32).to(device))
        l = loss(pred.squeeze(), batch_targets.to(torch.float32).to(device))
        print(f"\rloss = {l.item()}", end=" ")
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
    f1, accuracy = evaluate(neuralnet, my_dataloader_test, device)
    test_epoch.append(epoch)
    test_accuracy.append(accuracy)
    test_f1_score.append(f1)
    print(
        f"Test Accuracy at {epoch} : {accuracy} and F1 Score : {f1}"
    )
