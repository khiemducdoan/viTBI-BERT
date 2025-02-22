import torch
import transformers
from transformers import AutoTokenizer
model = torch.load(r"/media/data3/home/khiemdd/ViTBERT/src/best_model.pt",map_location=torch.device('cuda'))
model_name = "demdecuong/vihealthbert-base-syllable"
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
import os
import pathlib
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from captum.attr import IntegratedGradients, TokenReferenceBase, visualization
import numpy as np

class ViTBERT(Dataset):
    def __init__(self, data_path, tokenizer_name):
        self.dataset = pd.read_csv(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __getitem__(self, idx):
        text = self.dataset.iloc[idx, 0]  # Extract text from dataframe
        label = self.dataset.iloc[idx, 1] - 1.0  # Extract label and shift by 1.0

        # Tokenization
        tokenized_text = self.tokenizer(text, add_special_tokens=True, max_length= 50,return_tensors='pt', padding='max_length', truncation=True)

        # Extract fields from tokenized text
        input_ids = tokenized_text['input_ids'].squeeze().long()
        attention_mask = tokenized_text['attention_mask'].squeeze().long()

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)  # Ensure the label is a long tensor

        # Return tensors
        return input_ids, attention_mask, label, text  # Return attention_mask

    def __len__(self):
        return len(self.dataset)

data_path = r"/media/data3/home/khiemdd/ViTBERT/dataset/dataset_chi_ha_hieu/dataset_final_test_after.csv"
tokenizer_name = "demdecuong/vihealthbert-base-word"
model_name = "demdecuong/vihealthbert-base-word"

dataset = ViTBERT(data_path, tokenizer_name=tokenizer_name)
test_loader = DataLoader(dataset, batch_size=1)

model.eval()

# Fetch a single batch from the DataLoader
input_ids, attention_mask, label, text = next(iter(test_loader))

# Print details of the fetched batch
print("Input IDs:", input_ids)
print("Attention Mask:", attention_mask)
print("Label:", label)
print("Labels shape:", label.shape)
print("Input IDs Shape:", input_ids.shape)

# Initialize IntegratedGradients

import torch
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
device = "cuda"
# Assuming your model and dataset are defined as in your previous message

# Step 1: Initialize LIME explainer
explainer = LimeTextExplainer(class_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])

# Step 2: Define predict function for LIME
def predict(texts):
    inputs = tokenizer(texts, add_special_tokens=True, max_length=50, return_tensors='pt', padding='max_length', truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    logits = model(input_ids, attention_mask)
    probabilities = torch.softmax(logits, dim=1)
    return probabilities.detach().cpu().numpy()

# Step 3: Fetch a single instance (text) from DataLoade

# Step 4: Interpret the model prediction using LIME
explanation = explainer.explain_instance(text[0], predict, num_features=50, top_labels=1)

# Step 5: Print explanations
print('Text:', text[0])
print('True Label:', label.item())
print('Predicted Probability:', predict([text[0]]))