import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision.transforms import ToTensor
import time
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr, pearsonr
import torch.optim as optim
import numpy as np

torch.manual_seed(2025)
def get_device(use_gpu=True):
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Helper function to visualize performance during training (from homework 4)
def plot_training_curves(train_losses, val_losses):
    """Plot training loss and validation losses.
    
    Parameters
    ----------
    train_losses : list of float
        Training loss values for each epoch. Should have one value per epoch.
    val_losses : list of float
        Validation loss values for each epoch. Should have one value per epoch.
        
    Returns
    -------
    None
        Displays matplotlib figure with two subplots showing training curves.
        
    Examples
    --------
    >>> train_losses = [0.8, 0.6, 0.4, 0.3, 0.2]
    >>> val_accuracies = [0.75, 0.80, 0.85, 0.87, 0.88]
    >>> plot_training_curves(train_losses, val_accuracies)
    """
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.show()
    
def train_model(model, train_loader, val_loader, loss_fn, optimizer, patience=10, max_epochs=100): 
    #dataloader will be train_loader or val_loader
    device = get_device(True)
    start_time = time.time()
    train_losses = []
    val_losses = []
    patience_count = 0
    best_val_loss = float('inf')
        
    for epoch in range(max_epochs):
        model.train()
        train_loss_sum = 0.0
            
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * data.size(0)
                
        avg_train_loss = train_loss_sum / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
            
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = loss_fn(outputs, labels)
                val_loss_sum += loss.item() * data.size(0)
        avg_val_loss = val_loss_sum / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_count = 0
            best_weights = model.state_dict()
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
        if epoch % 10 == 0 or epoch == max_epochs - 1:
            print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
            
    model.load_state_dict(best_weights)                    
    training_time = time.time() - start_time
    
    plot_training_curves(train_losses, val_losses)
    
    return model, train_losses, val_losses, training_time


def test_model(model, test_loader):
    device = get_device(True)
    model.eval()  
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  
        for data, labels in test_loader:
            data = data.to(device)
            #labels = labels.to(device)
            outputs = model(data)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    r2 = r2_score(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    spearman_corr, _ = spearmanr(all_labels, all_preds)
    pearson_corr, _ = pearsonr(all_labels, all_preds)

    return {"R2": r2, "RMSE": rmse, "Spearman": spearman_corr, "Pearson": pearson_corr}, all_preds, all_labels

