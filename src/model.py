import torch
import torch.nn as nn
from transformers import AutoModel
from data_loader import ViTBERT
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
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, num_classes))
        layers.append(nn.Softmax(dim=1))
        
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
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.classifier = DynamicClassifier(input_dim=self.bert.config.hidden_size,
                                            num_classes=num_classes,
                                            dropout_rate=dropout_rate,
                                            num_layers=num_layers)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Access last_hidden_state (mandatory) and optionally pooler_output
        last_hidden_state = outputs.last_hidden_state  # Always available
        pooler_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else None

        # Use pooler_output if available; otherwise, use mean pooling
        if pooler_output is not None:
            logits = self.classifier(pooler_output)
        else:
            # Mean pooling over the last hidden state
            logits = self.classifier(last_hidden_state.mean(dim=1))
        
        return logits


# To use your modified classifier, you just need to specify the desired number of layers and dropout rate when initializing it:
def main():
    data_path = "/media/data3/home/khiemdd/ViTBERT/dataset/donedataset_test_after_low.csv"
    tokenizer = "demdecuong/vihealthbert-base-syllable"
    

    dataset = ViTBERT(data_path, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    pretrained_model_name = "demdecuong/vihealthbert-base-syllable"  # change according to your model
    model = ViTBERTClassifier(pretrained_model_name=pretrained_model_name,
                              num_classes=4,
                              dropout_rate=0.25,
                              num_layers=4)

    # Suppose you have a CUDA-enabled GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    # Train for one epoch
    
    model.train()
    batch = next(iter(dataloader))
    input_ids, attention_mask, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
    labels = labels.long()
    print("labels: ")
    print(labels)
    print("shape of labels: ")
    print(labels.shape)
    outputs = model(input_ids, attention_mask)
    print("output: ")
    print(outputs)
    print("output shape: ")
    print(outputs.shape)
    ground_truth_class_ids = outputs.argmax(axis=1)
    print("ground_truth_class_ids shape: ")
    print(ground_truth_class_ids.shape)
    print("ground_truth_class_ids: ")
    print(ground_truth_class_ids)
    outputs = outputs.float()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    preds = outputs.argmax(dim=1)
    print(preds.shape)
    print(preds)
        
    print("Output size:", outputs.size())
    print("Output type:", type(outputs))
    print("Loss value:", loss.item())
    print(model.bert.embeddings)
if __name__ == "__main__":
    main()

