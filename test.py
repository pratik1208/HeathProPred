import torch
from models import *
import pandas as pd

file_path = "/home2/pratik_2211ai19/TechGIG/train.py"

neuralnet = NeuralNetwork()
neuralnet.load_state_dict(torch.load(file_path))

neuralnet.eval()

model_parameters = neuralnet.parameters()

for params in model_parameters:
    print(params)

dictonary = {}
dictonary["DEVICETYPE"] = input()
dictonary["PLATFORM_ID"] = input()
dictonary["BIDREQUESTIP"] = input()
dictonary["USERPLATFORMUID"] = input()
dictonary["USERCITY"] = input()
dictonary["USERZIPCODE"] = input()
dictonary["USERAGENT"] = input()
dictonary["PLATFORMTYPE"] = input()
dictonary["CHANNELTYPE"] = input()
dictonary["URL"] = input()
dictonary["KEYWORDS"] = input()
dictonary["TAXONOMY"] = input()

print(dictonary)
df = pd.DataFrame.from_dict(dictonary, orient='index', columns= dictonary.keys())

neuralnet.predict()
