import data_loader
import trainer 
from trainer_without_sweep import ViTBERTTrainer
import utils
import yaml
import wandb
import random
import torch
import numpy as np
from utils import load_config
if __name__ == "__main__":
    # Specify the path to your YAML config file
    print("==========CONCAT=============")
    config_path = '/media/data3/home/khiemdd/ViTBERT/src/config_without_sweep.yaml'  # Update this path
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    # Load the configuration
    configfile = load_config(config)
    
    # Initialize the trainer with the loaded configuration
    trainer = ViTBERTTrainer(configfile)
    
    # Start the training process
    trainer.train() 
    #trainer.sweep()