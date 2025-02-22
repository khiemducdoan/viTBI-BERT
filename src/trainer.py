import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
from tqdm.auto import tqdm
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_fscore_support

from data_loader import ViTBERT # Adjust the import statement according to your project structure
from model import ViTBERTClassifier # Adjust the import statement according to your project structure
from utils import get_device # Assuming you have a utility function to determine the device

class ViTBERTTrainer:
    def __init__(self, config):
        self.model_save_path = config.model_save_path
        self.patience = config.train.early_stopping_patience  # Number of epochs to wait after last time validation loss improved.
        self.best_val_loss = float('inf')  # Initialize best validation loss to infinity
        self.epochs_without_improvement = 0  # Counter for epochs without improvement
        
        self.device = get_device()
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.lr = config.train.lr
        self.lr_min = config.train.lr_min
        self.lr_max = config.train.lr_max
        self.epochs = config.train.epochs
        self.train_path = config.path.train_path
        self.test_path = config.path.test_path
        self.num_output = config.data.num_output
        self.pretrained_model_name = config.model.pretrained_name
        self.date = config.train.date
        self.dropout_rate = config.train.dropout_rate
        self.num_layer = config.train.num_layer
        self.num_sweep = config.train.num_sweep
        self.project_name = config.train.project_name
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
        
        self.sweep_configuration = {
            "method": "random",
            "name": self.project_name,
            "metric": {"goal": "maximize", 
                       "name": "highest accuracy"},
            "parameters": {
                "batch_size": {"values": self.batch_size},
                "lr": {"max": self.lr_max, "min": self.lr_min},
                "dropout_rate": {"values": self.dropout_rate},
                "num_layer": {"values": self.num_layer},
                "pretrained_model_name": {"values": self.pretrained_model_name}
            },
        }
        self.login_loger()
        
        # Load checkpoint if specified
    def get_model_name(self,config):
        if config.pretrained_model_name == "demdecuong/vihealthbert-base-word":
            return "demdecuong/vihealthbert-base-word"
        else:
            return "demdecuong/vihealthbert-base-syllable"
    def login_loger(self):
        self.sweep_id = wandb.sweep(sweep=self.sweep_configuration, project="ViTBERT")
    def _create_dataloader(self, path,mode,model_name,config):
        dataset = ViTBERT(data_path=path,tokenizer=model_name) # Ensure ViTBERTDataset is implemented
        return DataLoader(dataset=dataset, batch_size=config.batch_size, num_workers=self.num_workers, shuffle=True if mode == "train" else False)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        train_acc = 0
        for batch in tqdm(dataloader, desc="Training"):
            input_ids,attention_mask ,labels,text = batch[0].to(self.device),batch[1].to(self.device),batch[2].to(self.device),batch[3]
            labels = labels.long()
            self.optimizer.zero_grad()
            outputs = self.model(input_ids,attention_mask)
            outputs = outputs.float()
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            y_pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            train_acc += (y_pred_class == labels).sum().item()/len(outputs)
        return total_loss / len(dataloader),train_acc / len(dataloader)
    def validate_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_labels = []
        all_preds = []
        data_false = {}

        with torch.inference_mode():
            for batch in tqdm(dataloader, desc="Validation"):
                input_ids, attention_mask, labels, text = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device), batch[3]
                labels = labels.long()
                outputs = self.model(input_ids, attention_mask)
                outputs = outputs.float()
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                preds = outputs.argmax(dim=1)

                all_labels.extend(labels.cpu().detach().numpy())  # Collect labels
                all_preds.extend(preds.cpu().detach().numpy())  # Collect predictions

                correct_preds = (preds == labels).sum().item()
                total_correct += correct_preds
                total_samples += len(labels)

                # Collect incorrect predictions
                for i in range(len(labels)):
                    if preds[i] != labels[i]:
                        data_false[text[i]] = {
                            "label": labels[i].cpu().numpy().tolist(),
                            "pred": preds[i].cpu().numpy().tolist()
                        }

            # Calculate metrics after iterating through all batches
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

            # Log Metrics
            wandb.log({"precision": precision})
            wandb.log({"recall": recall})
            wandb.log({"f1_score": f1})

            metrics = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "loss": total_loss / len(dataloader),
                "accuracy": total_correct / total_samples,
                "all_things": [all_labels, all_preds],
                "data_false": data_false
            }

        return metrics

    def train(self,config = None):
        with wandb.init(config = config ):
            config = wandb.config
            pretrained_model_name = self.get_model_name(config)
            self.train_loader = self._create_dataloader(self.train_path, "train",model_name= pretrained_model_name,config = config)
            self.test_loader = self._create_dataloader(self.test_path, "test",model_name= pretrained_model_name, config = config)
            self.model = ViTBERTClassifier(num_classes=self.num_output,
                                           pretrained_model_name=pretrained_model_name,
                                           dropout_rate= config.dropout_rate,
                                           num_layers= config.num_layer
                                           ).to(self.device)
            self.optimizer = Adam(self.model.parameters(), lr=config.lr)
            epoch_arr = []
            train_accs = []
            test_accs = []
            train_losses = []
            test_losses = []
            best_metrics = {}
            max_test_accuracy = 0
            for epoch in range(self.epochs):
                epoch_arr.append(epoch)
                train_loss,train_acc = self.train_epoch(self.train_loader)
                train_accs.append(train_acc)
                train_losses.append(train_loss)
                metrics = self.validate_epoch(self.test_loader)
                val_loss = metrics["loss"]
                val_acc = metrics["accuracy"]
                test_accs.append(metrics["accuracy"])
                test_losses.append(metrics["loss"])
                
                if val_acc > max_test_accuracy:
                    max_test_accuracy = val_acc
                    best_metrics = metrics
                    best_test_accs = test_accs
                    best_test_losses = test_losses
                    best_train_accs = train_accs
                    best_train_losses = train_losses
                wandb.log({"highest accuracy": max_test_accuracy})    
                
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
                print(f"Epoch {epoch+1}/{self.epochs}, Train accuracy: {train_acc:.4f}, Validation accuracy: {val_acc:.4f}")
                wandb.log({"train loss": train_loss, "train accuracy": train_acc, "test loss" : metrics["loss"], "test accuracy": metrics["accuracy"]})
            wandb.log({"conf_mat1_for_best_epoch" : wandb.plot.confusion_matrix(probs = None,y_true=best_metrics["all_things"][0], preds=best_metrics["all_things"][1], class_names=["1","2","3","4"])})
            wandb.log({"conf_mat2_for_best_epoch" : wandb.sklearn.plot_confusion_matrix(best_metrics["all_things"][0], best_metrics["all_things"][1], ["1","2","3","4"])})
            wandb.log({"RESULTS: ": wandb.Table(columns=["precision", "recall", "f1_score", "accuracy"], 
                                                data=[[best_metrics["precision"],best_metrics["recall"],best_metrics["f1_score"],best_metrics["accuracy"]]])})
            wandb.log(
            {
                "train vs. test accuracy": wandb.plot.line_series(
                    xs=epoch_arr,
                    ys=[best_train_accs, best_test_accs],
                    keys=["Train accuracy", "Test accuracy"],
                    title="Train vs. Test Accuracy",
                )
            })
            wandb.log(
            {
                "train vs. test loss": wandb.plot.line_series(
                    xs=epoch_arr,
                    ys=[best_train_losses, best_test_losses],
                    keys=["Train loss", "Test loss"],
                    title="Train vs. Test Loss",
                )
            })
            data_false_table = wandb.Table(columns=["Text", "Predicted Label", "True Label"])
            for text, errors in best_metrics["data_false"].items():
                data_false_table.add_data(text, errors["pred"], errors["label"])
            wandb.log({"Data False Table": data_false_table})

    def sweep(self):
        wandb.agent(self.sweep_id, function=self.train, count=self.num_sweep)
