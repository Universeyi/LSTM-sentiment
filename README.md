# LSTM with Multi-head Attention for Sentiment Analysis

## About

This repository contains the official implementation of the paper:

> **Advancing Sentiment Analysis: A Novel LSTM Framework with Multi-head Attention**  
> Jingyuan Yi, Peiyang Yu, Tianyi Huang, Xiaochuan Xu  
> arXiv:2503.08079 [cs.CL]  
> [https://arxiv.org/abs/2503.08079](https://arxiv.org/abs/2503.08079)

If you find this code useful in your research, please cite:

```bibtex
@misc{yi2025advancingsentimentanalysisnovel,
      title={Advancing Sentiment Analysis: A Novel LSTM Framework with Multi-head Attention}, 
      author={Jingyuan Yi and Peiyang Yu and Tianyi Huang and Xiaochuan Xu},
      year={2025},
      eprint={2503.08079},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.08079}, 
}
```

## Overview

This project implements a sentiment analysis model based on LSTM and multi-head attention mechanism. By integrating TF-IDF feature extraction and multi-head attention, the model significantly improves the performance of text sentiment analysis.

## Model Architecture

![Network Structure](network_structure.svg)

The model consists of the following key components:

1. **Input Layer**: Receives batch input features (batch_size, input_dim)
2. **LSTM Layer**: Processes sequence data using LSTM with hidden dimension of 10
3. **Multi-head Attention Layer**: Implements single-head attention mechanism to enhance focus on important features
4. **Fully Connected Output Layer**: Maps features to final classification results

## Key Features

- Integration of TF-IDF feature extraction
- Enhanced feature learning through multi-head attention mechanism
- Approximately 12% accuracy improvement over standard LSTM models
- Achieved 80.28% accuracy on the test set

## Performance Advantages

- Significant improvements in accuracy, recall, and F1 score compared to baseline models
- Ablation experiments demonstrate the necessity of all modules
- Multi-head attention mechanism contributes most to performance improvement

## Applications

- Public opinion monitoring
- Product recommendation systems
- User sentiment analysis
- Text classification tasks

## Usage Guide

### Requirements

```
torch
pandas
numpy
scikit-learn
matplotlib
seaborn
graphviz
```

### Running Instructions

1. Data Preparation:
   ```python
   # Data should be saved in Excel format with labels in the last column
   data = pd.read_excel('output.xlsx')
   ```

2. Training the Model:
   ```python
   python lstm_pytorch.py
   ```

3. Output Results:
   - Best model weights are saved to `best_model.pth`
   - Training process plots are saved as `training_process.png`
   - Detailed logs are saved in `training_log_[timestamp].txt`

## Experimental Results

### Model Architecture Visualization
![Network Structure](network_structure.svg)

### Training and Validation Performance
![Training Process](training_process.png)
The training process plot shows the model's learning curve, including:
- Training and validation loss trends
- Training and validation accuracy progression
- Model convergence characteristics

### Classification Performance
![Confusion Matrix](confusion_matrix.png)
The confusion matrix demonstrates the model's classification performance across different categories.

### ROC Analysis
![ROC Curves](roc_curves.png)
The ROC curves show:
- Individual class performance
- Multi-class classification capability
- Area Under Curve (AUC) metrics

Key performance metrics:
- Overall Accuracy: 80.28%
- Significant improvement over baseline: ~12%
- Strong performance across all classes as shown in ROC curves

## Visualization

The model provides comprehensive visualization tools for model analysis:

- Network Structure: Automatically generates network architecture diagram (`network_structure.svg`)
- Model Performance: 
  - Confusion matrix visualization (`confusion_matrix.png`)
  - ROC curves for multi-class classification (`roc_curves.png`)
  - Training and validation metrics plots (`training_process.png`)
  - Detailed training logs (`training_log_[timestamp].txt`)

All visualization results are automatically saved during the training process. You can find these files in the project root directory after running `lstm_pytorch.py`.


## License

MIT License 