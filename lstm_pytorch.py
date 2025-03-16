import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
import random
import time
from graphviz import Digraph
from datetime import datetime
from IPython.display import display, SVG

# Set random seed to ensure reproducibility
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# LSTM model definition
class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super(LSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # Adjust input dimensions to (batch_size, sequence_length, input_dim)
        x = x.unsqueeze(1)
        
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the output of the last time step
        out = attn_output[:, -1, :]
        
        # Fully connected layer
        out = self.fc(out)
        return out

# Training function
def train_model(model: nn.Module, 
                train_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device) -> float:
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return total_loss / len(train_loader), accuracy

# Evaluation function
def evaluate_model(model: nn.Module,
                  test_loader: DataLoader,
                  criterion: nn.Module,
                  device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            total_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = 100 * correct / total
    return total_loss / len(test_loader), accuracy, np.array(all_preds), np.array(all_labels), np.array(all_probs)

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Plot ROC curve
def plot_roc_curve(y_true, y_pred_proba, title):
    """
    Plot multi-class ROC curves
    y_true: true labels
    y_pred_proba: model's probability predictions
    """
    plt.figure(figsize=(10, 8))
    
    # Get number of classes
    n_classes = len(np.unique(y_true))
    
    # Calculate ROC curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Convert to one-hot encoding
    y_true_bin = np.eye(n_classes)[y_true]
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2,
                label=f'ROC curve of class {i} (AUC = {roc_auc[i]:0.2f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    
    # Return average AUC
    return np.mean(list(roc_auc.values()))

def create_network_diagram(input_dim: int, hidden_dim: int, num_classes: int):
    """Create and display network structure diagram"""
    # Create a directed graph
    dot = Digraph(comment='LSTM with Attention Network Structure')
    dot.attr(rankdir='LR')  # Left to right layout
    
    # Set node style
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    
    # Add input layer
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Input Layer')
        c.attr('node', shape='box', style='rounded,filled', fillcolor='lightgreen')
        c.node('input', f'Input Features\n(batch_size, {input_dim})')
    
    # Add LSTM layer
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='LSTM Layer')
        c.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
        c.node('lstm', f'LSTM\n(hidden_dim={hidden_dim})')
    
    # Add attention layer
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='Attention Layer')
        c.attr('node', shape='box', style='rounded,filled', fillcolor='lightyellow')
        c.node('attention', 'Multi-head Attention\n(num_heads=1)')
    
    # Add fully connected layer
    with dot.subgraph(name='cluster_3') as c:
        c.attr(label='Output Layer')
        c.attr('node', shape='box', style='rounded,filled', fillcolor='lightpink')
        c.node('fc', f'Fully Connected\n(num_classes={num_classes})')
    
    # Add edges
    dot.edge('input', 'lstm', 'reshape\n(batch_size, 1, input_dim)')
    dot.edge('lstm', 'attention', 'LSTM output\n(batch_size, 1, hidden_dim)')
    dot.edge('attention', 'fc', 'attention output\n(batch_size, hidden_dim)')
    
    # Save and display diagram
    dot.render('network_structure', format='svg', cleanup=True)
    
    # Display SVG in Colab
    display(SVG('network_structure.svg'))

def format_time(seconds):
    """Format time in HH:MM:SS"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    set_seed(42)
    
    # Create log file
    log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    def log_message(message):
        """Log message to file and print to console"""
        print(message)
        with open(log_filename, 'a') as f:
            f.write(message + '\n')
    
    log_message("Starting data loading...")
    
    # Load data
    data = pd.read_excel('output.xlsx')
    
    # Prepare features and labels
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values - 1  # Ensure labels start from 0
    
    log_message(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
    log_message(f"Number of classes: {len(np.unique(y))}")
    
    # Data normalization
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # Split training and test sets
    num_samples = len(X)
    train_size = int(0.7 * num_samples)
    
    # Shuffle data
    indices = np.random.permutation(num_samples)
    X = X[indices]
    y = y[indices]
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    log_message(f"Training set size: {len(X_train)} samples")
    log_message(f"Test set size: {len(X_test)} samples")
    
    # Create data loaders
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Model parameters
    input_dim = X.shape[1]
    hidden_dim = 10
    num_classes = len(np.unique(y))
    
    # Visualize network structure
    log_message("Generating network structure diagram...")
    create_network_diagram(input_dim, hidden_dim, num_classes)
    log_message("Network structure diagram generated")
    
    # Create model
    model = LSTMWithAttention(input_dim, hidden_dim, num_classes).to(device)
    log_message("\nModel structure:")
    log_message(str(model))
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_message(f"\nTotal parameters: {total_params:,}")
    log_message(f"Trainable parameters: {trainable_params:,}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    num_epochs = 150
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    log_message("\nStarting training...")
    start_time = time.time()
    best_test_acc = 0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _, _ = evaluate_model(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        remaining_time = (epoch_time * (num_epochs - epoch - 1))
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        if (epoch + 1) % 10 == 0:
            log_message(
                f'Epoch [{epoch+1}/{num_epochs}] '
                f'Time: {format_time(epoch_time)} '
                f'Total: {format_time(total_time)} '
                f'ETA: {format_time(remaining_time)}\n'
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% '
                f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% '
                f'Best Test Acc: {best_test_acc:.2f}%'
            )
    
    total_time = time.time() - start_time
    log_message(f"\nTraining completed! Total time: {format_time(total_time)}")
    log_message(f"Best test accuracy: {best_test_acc:.2f}%")
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate final model
    _, _, test_preds, test_labels, test_probs = evaluate_model(model, test_loader, criterion, device)
    
    # Plot confusion matrix
    log_message("\nGenerating confusion matrix...")
    plot_confusion_matrix(test_labels, test_preds, 'Confusion Matrix for Test Data')
    
    # Plot ROC curves
    log_message("Generating ROC curves...")
    roc_auc = plot_roc_curve(test_labels, test_probs, 'ROC Curves for Test Data')
    log_message(f"Average ROC AUC: {roc_auc:.4f}")
    
    # Plot training process
    log_message("Generating training process plots...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_process.png')
    plt.show()
    
    log_message("All results have been saved")

if __name__ == "__main__":
    main() 