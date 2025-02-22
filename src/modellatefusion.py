import torch
import torch.nn as nn
from transformers import AutoModel
from data_loaderlatefusion import ViTBERT
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
class DynamicClassifier(nn.Module):
    def __init__(self, 
                input_dim=768, 
                num_classes=4, 
                dropout_rate=0.25, 
                num_layers=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            output_dim = input_dim // 2
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.GELU())
            if i < num_layers - 1:  # No dropout after last layer before classification
                layers.append(nn.Dropout(dropout_rate))
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, feature):
        return self.model(feature)

class ViTBERTClassifier(nn.Module):
    def __init__(self, 
                pretrained_model_name, 
                num_classes=4, 
                dropout_rate=0.25, 
                num_layers=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_classes, hidden_size=64, num_layers=1, batch_first=True)
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.classifier1 = DynamicClassifier(input_dim=self.bert.config.hidden_size,
                                            num_classes=num_classes,
                                            dropout_rate=dropout_rate,
                                            num_layers=num_layers)
        self.classifier2 = DynamicClassifier(input_dim=self.bert.config.hidden_size,
                                            num_classes=num_classes,
                                            dropout_rate=dropout_rate,
                                            num_layers=num_layers)
        self.final_classifier = nn.Linear(64, num_classes)
    def forward(self, input_ids1, input_ids2,attention_mask1=None, attention_mask2=None):
        outputs1 = self.bert(input_ids=input_ids1, attention_mask=attention_mask1)
        last_hidden_state1 = outputs1[1]
        
        outputs2 = self.bert(input_ids=input_ids2, attention_mask=attention_mask2)
        last_hidden_state2 = outputs2[1]
        
        logits1 = self.classifier1(last_hidden_state1)
        logits2 = self.classifier2(last_hidden_state2)
        logits = logits1 + logits2
        return nn.Softmax(dim=1)(logits)
    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

# To use your modified classifier, you just need to specify the desired number of layers and dropout rate when initializing it:
def main():
    # Path to your data
    data_path = "/media/data3/home/khiemdd/ViTBERT/dataset/data500/DATASET_ALL.csv"
    
    # Tokenizer
    tokenizer_name = "demdecuong/vihealthbert-base-syllable"
    
    # Load dataset
    dataset = ViTBERT(data_path=data_path,
                                      stop_words_file= "/media/data3/home/khiemdd/ViTBERT/dataset/needed_files/vietnamese-stopwords.txt",
                                      wordnet_file= "/media/data3/home/khiemdd/ViTBERT/dataset/needed_files/word_net_vi.json",
                                      indices= None,
                                      type = "test",
                                      tokenizer=tokenizer_name)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize the model
    pretrained_model_name = tokenizer_name  # Use the same model as the tokenizer
    model = ViTBERTClassifier(pretrained_model_name=tokenizer_name,
                              num_classes=4,
                              dropout_rate=0.25,
                              num_layers=4)

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Get a single batch
    batch = next(iter(dataloader))
    
    # Move data to the appropriate device
    input_ids1, attention_mask1, input_ids2, attention_mask2, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)
    labels = labels.long()

    # Print the shapes
    print("Shape of input_ids1:", input_ids1.shape)
    print("Shape of attention_mask1:", attention_mask1.shape)
    print("Shape of input_ids2:", input_ids2.shape)
    print("Shape of attention_mask2:", attention_mask2.shape)
    print("Shape of labels:", labels.shape)
    print(labels)
    # Forward pass to check output shapes
    outputs = model(input_ids1=input_ids1, input_ids2=input_ids2, attention_mask1=attention_mask1, attention_mask2=attention_mask2)
    
    # Print output shapes
    print("Shape of model outputs:", outputs.shape)
    
    # Check if predictions are computed correctly
    preds = outputs.argmax(dim=1)
    print("Shape of preds:", preds.shape)
    print("Shape of labels:", labels.shape)
    print(outputs)
    print(preds)
    
if __name__ == "__main__":
    main()