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
import random
from sklearn.model_selection import KFold,StratifiedKFold 

from data_loader import ViTBERT # Adjust the import statement according to your project structure
from model import ViTBERTClassifier # Adjust the import statement according to your project structure
from utils import get_device # Assuming you have a utility function to determine the device

class ViTBERTTrainer:
    def __init__(self, config):
        self.stop_words_file = config.data.stop_words_file
        self.wordnet_file = config.data.wordnet_file
        self.patience = config.train.early_stopping_patience  # Number of epochs to wait after last time validation loss improved.
        self.best_val_loss = float('inf')  # Initialize best validation loss to infinity
        self.epochs_without_improvement = 0  # Counter for epochs without improvement
        self.k_folds = config.train.k_folds
        self.model_save_path = config.model_save_path
        self.device = get_device()
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.lr = config.train.lr
        self.epochs = config.train.epochs
        self.train_path = config.path.train_path
        self.test_path = config.path.test_path
        self.dropout_rate = config.train.dropout_rate
        self.num_layers = config.train.num_layers
        self.pretrained_model_name = config.model.pretrained_name
        self.dataset = ViTBERT(data_path=self.train_path,
                                      stop_words_file= self.stop_words_file,
                                      wordnet_file= self.wordnet_file,
                                      indices= None,
                                      type = "test",
                                      tokenizer=self.pretrained_model_name) # Ensure ViTBERTDataset is implemented
        self.num_output = config.data.num_output

        self.date = config.train.date
        self.criterion = torch.nn.CrossEntropyLoss()
        self.login_loger()
        # Load checkpoint if specified
    def login_loger(self):
        wandb.login()
        wandb.init(project='ViTBERT', config={
            "learning_rate" : self.lr,
            "architecture": "demdecuong/vihealthbert-base-syllable",
            "dataset": "acronym processed",
            "epoch": 30,
            "batch_size": self.batch_size
        })
    def set_all_seed(self):
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
        np.random.seed(seed)
        random.seed(seed)
    def reset_weights(self, m):
        '''
        Try resetting model weights to avoid
        weight leakage.
        '''
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                print(f'Reset trainable parameters of layer = {layer}')
                # Check if the parameters are trainable
                for param in layer.parameters():
                    if param.requires_grad:
                        layer.reset_parameters()
                        break
    def _create_dataloader(self, path,mode, sampler):
        dataset = ViTBERT(data_path=path,tokenizer=self.pretrained_model_name) # Ensure ViTBERTDataset is implemented
        return DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True if mode == "train" else False, sampler= sampler)

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
    def train(self):
        self.set_all_seed()
        kfold = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state= 42)
        best_fold_metrics = []
        best_overall_accuracy = 0
        best_model = None
        labels = np.array([self.dataset[i][2] for i in range(len(self.dataset))])
        print('---------------------------------------------------------------------------------------------')
        for fold, (train_ids, test_ids) in enumerate(kfold.split(np.zeros(len(self.dataset)), labels)):
            print(f'FOLD {fold}')
            print('--------------------------------')
            # self.train_data = ViTBERT(data_path=self.train_path,
            #                           stop_words_file= self.stop_words_file,
            #                           wordnet_file= self.wordnet_file,
            #                           indices= train_ids,
            #                           type = "train",
            #                           tokenizer=self.pretrained_model_name) # Ensure ViTBERTDataset is implemented
            # self.test_data = ViTBERT(data_path=self.train_path,
            #                           stop_words_file= self.stop_words_file,
            #                           wordnet_file= self.wordnet_file,
            #                           indices= test_ids,
            #                           type = "test",
            #                           tokenizer=self.pretrained_model_name) # Ensure ViTBERTDataset is implemented
            
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size = self.batch_size, sampler=train_subsampler)
            self.test_loader = torch.utils.data.DataLoader(self.dataset, batch_size = self.batch_size,sampler=test_subsampler)
            epoch_arr = []
            train_accs = []
            test_accs = []
            train_losses = []
            test_losses = []
            
            best_metrics = {}
            max_test_accuracy = 0

            self.model = ViTBERTClassifier(num_classes=self.num_output,
                                           pretrained_model_name=self.pretrained_model_name,
                                           dropout_rate= self.dropout_rate,
                                           num_layers= self.num_layers
                                           ).to(self.device)
            self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
            # self.model.apply(self.reset_weights)
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
                    # best_model_params = self.model.state_dict()
                    best_test_accs = test_accs
                    best_test_losses = test_losses
                    best_train_accs = train_accs
                    best_train_losses = train_losses
                    torch.save(self.model, f'./best-model-fold-{fold}.pt')
                    
                
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
                print(f"Epoch {epoch+1}/{self.epochs}, Train accuracy: {train_acc:.4f}, Validation accuracy: {val_acc:.4f}")
                wandb.log({f"train loss of fold {str(fold)}": train_loss, f"train accuracy fold {fold}": train_acc, f"test loss fold {fold}" : metrics["loss"], f"test accuracy fold {fold}": metrics["accuracy"]})
            best_fold_metrics.append({
                'fold': fold,
                'accuracy': best_metrics["accuracy"],
                'precision': best_metrics["precision"],
                'recall': best_metrics["recall"],
                'f1_score': best_metrics["f1_score"]
            })
            wandb.log({f"conf_mat1_for_best_epoch fold {fold} " : wandb.plot.confusion_matrix(probs = None,y_true=best_metrics["all_things"][0], preds=best_metrics["all_things"][1],title = f"ConfusionMatrix for fold {fold}", class_names=["1","2","3","4"])})
            wandb.log({f"conf_mat2_for_best_epoch {fold}" : wandb.sklearn.plot_confusion_matrix(best_metrics["all_things"][0], best_metrics["all_things"][1], ["1","2","3","4"])})
            wandb.log({f"RESULTS fold {fold}: ": wandb.Table(columns=["precision", "recall", "f1_score", "accuracy"], 
                                                data=[[best_metrics["precision"],best_metrics["recall"],best_metrics["f1_score"],best_metrics["accuracy"]]])})
            wandb.log(
            {
                f"train vs. test accuracy fold {fold}": wandb.plot.line_series(
                    xs=epoch_arr,
                    ys=[best_train_accs, best_test_accs],
                    keys=["Train accuracy", "Test accuracy"],
                    title="Train vs. Test Accuracy",
                )
            })
            wandb.log(
            {
                f"train vs. test loss fold {fold}": wandb.plot.line_series(
                    xs=epoch_arr,
                    ys=[best_train_losses, best_test_losses],
                    keys=["Train loss", "Test loss"],
                    title=f"Train vs. Test Loss {fold}",
                )
            })
            data_false_table = wandb.Table(columns=["Text", "Predicted Label", "True Label"])
            for text, errors in best_metrics["data_false"].items():
                data_false_table.add_data(text, errors["pred"], errors["label"])
            wandb.log({f"Data False Table fold {fold}": data_false_table})
        # save_path = f'./after-model-fold-{best_model_fold}.pt'
        # torch.save(best_model, save_path)
        print('\nBest Metrics for Each Fold:')
        print('Fold | Accuracy | Precision | Recall | F1 Score')
        print('-----------------------------------------------')
        for metrics in best_fold_metrics:
            print(f'{metrics["fold"]:^4} | {metrics["accuracy"]:.4f}  | {metrics["precision"]:.4f}  | {metrics["recall"]:.4f}  | {metrics["f1_score"]:.4f}')