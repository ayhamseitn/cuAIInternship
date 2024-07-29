import pandas as pd
import numpy as np
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

# Load the JSON data
coqa = pd.read_json("C:/Users/Dell/Desktop/Staj/cuAI01/data1son.json")
coqa.head()

# Required columns in our dataframe
cols = ["text", "question", "answer"]

# List of lists to create our dataframe
comp_list = []
for index, row in coqa.iterrows():
    for i in range(len(row["data"]["questions"])):
        temp_list = []
        temp_list.append(row["data"]["story"])
        temp_list.append(row["data"]["questions"][i]["input_text"])
        temp_list.append(row["data"]["answers"][i]["input_text"])
        comp_list.append(temp_list)

new_df = pd.DataFrame(comp_list, columns=cols)

# Saving the dataframe to CSV file with '/' as delimiter
new_df.to_csv("C:/Users/Dell/Desktop/Staj/cuAI01/data1son.csv", sep='|', index=False)

# Loading the CSV file with '/' as delimiter
data = pd.read_csv("C:/Users/Dell/Desktop/Staj/cuAI01/data1son.csv", sep='|')
data.head()
