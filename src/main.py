import data_loader
import trainer 
from trainer import ViTBERTTrainer
import utils
import yaml
import wandb
from utils import load_config
import torch
import numpy as np 
import torch
import random

torch.cuda.empty_cache()
if __name__ == "__main__":
    # Specify the path to your YAML config file
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    config_path = '/media/data3/home/khiemdd/ViTBERT/src/config.yaml'  # Update this path
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    # Load the configuration
    configfile = load_config(config)
    
    # Initialize the trainer with the loaded configuration
    trainer = ViTBERTTrainer(configfile)
    
    # Start the training process
    #trainer.train()
    trainer.sweep()