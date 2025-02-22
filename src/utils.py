import os
import pathlib
import torch
import argparse
from pathlib import Path
import yaml

def load_config(d):
    namespace = argparse.Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, load_config(value))
        else:
            setattr(namespace, key, value)
    return namespace


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")
    
    
def get_class(data_dir):
    classes = sorted(entry.name for entry in os.scandir(data_dir) if entry.is_dir())
    class_to_idx = {class_name:i for i,class_name in enumerate(classes)}
    return classes, class_to_idx

def get_data_list(data_dir, range_index):
    data_list = []
    classes = sorted(entry.name for entry in os.scandir(data_dir) if entry.is_dir())
    for classe in classes:
        for i in range(range_index[0], range_index[1]):
            data_path = Path(os.path.join(data_dir, classe, f"{classe}{i}.jpg"))
            if data_path.exists():
                data_list.append(data_path)
    return data_list