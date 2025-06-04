# ViTBI-BERT: Vietnamese Traumatic Brain Injury Analysis with BERT

## Overview
ViTBI-BERT is a transformer-based model for analyzing Vietnamese medical texts related to Traumatic Brain Injury (TBI). It implements a multi-class classification system to categorize TBI severity levels using clinical notes and medical records.

## Features
- Pre-trained Vietnamese HealthBERT for medical text analysis
- Support for both word and syllable tokenization
- K-fold cross-validation for robust model evaluation 
- Data augmentation techniques for Vietnamese medical text
- Integrated attention visualization
- Model interpretability with Integrated Gradients
- WandB integration for experiment tracking
- Early stopping and learning rate scheduling

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- CUDA-enabled GPU (recommended)

### Setup
```bash
# Clone repository
git clone https://github.com/username/viTBI-BERT.git
cd viTBI-BERT

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preparation
```python
# Prepare k-fold datasets
python kfolddata.ipynb

# Augment training data (optional)
python data_custom.ipynb
```

### Training
```bash
# Basic training
python src/main_without_sweep.py

# Training with hyperparameter sweep
python src/main.py 

# Late fusion model training
python src/trainer_without_sweeplatefusion.py
```

## Data
The model works with Vietnamese medical text data:
- Input format: CSV with text and label columns
- 4 severity classes for TBI classification
- Support for both pre-procedure and post-procedure notes
- K-fold splits available in `dataset/datakfold/`

## Model

### Architecture
- Base model: ViHealthBERT 
  - Word-based: `demdecuong/vihealthbert-base-word`
  - Syllable-based: `demdecuong/vihealthbert-base-syllable`
- Custom classification head with configurable layers
- Late fusion capability for multiple text inputs

### Configuration
Model parameters can be configured in:
- `src/config_without_sweep.yaml` - Basic configuration
- `src/config.yaml` - Sweep configuration

## Training

### Parameters
- Learning rate: Configurable range (default: 1e-5 to 1e-3)
- Batch size: 1-32
- Dropout rate: 0.1-0.4 
- Number of layers: 1-3
- Early stopping patience: 6 epochs

### Features
- K-fold cross-validation
- Learning rate scheduling
- Early stopping
- Gradient clipping
- WandB experiment tracking

## Evaluation

### Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Analysis Tools
- Attention visualization (`attention_view.ipynb`)
- Integrated Gradients (`IG.ipynb`)
- Error analysis via WandB
- Custom performance plots

## Citation
```bibtex
@article{doan2024vitbi,
  title={ViTBI-BERT: A Vietnamese Language Model for Prediction of Traumatic Brain Injury},
  author={Doan, Duc-Khiem and Tran, Thanh-Hai and Tran, Trung-Kien and Le, Thi-Lan and Vu, Hai and Nguyen, Huu-Khanh and Can, Van-Mao and Nguyen, Thanh-Bac},
  journal={MIC-ICT Research},
  volume={2024},
  number={2},
  pages={1303},
  year={2024},
  doi={https://doi.org/10.32913/mic-ict-research.v2024.n2.1303},
  publisher={Military Institute of Science and Technology}
}

```

## License
This project is licensed under the Apache License 2.0.