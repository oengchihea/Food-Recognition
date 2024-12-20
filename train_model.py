import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from model import FoodClassifier
from dataset import create_data_loaders
from torch.nn.functional import softmax
from scipy.optimize import minimize

def mixup_data(x, y, alpha=0.2, use_cuda=False):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def find_optimal_temperature(logits, labels):
    """Find the optimal temperature for calibration."""
    def objective(temperature):
        scaled_logits = logits / temperature
        probs = softmax(torch.tensor(scaled_logits), dim=1).numpy()
        log_likelihood = np.mean(np.log(probs[range(len(labels)), labels] + 1e-12))
        return -log_likelihood
    
    result = minimize(objective, x0=1.5, method='nelder-mead')
    return result.x[0]

def train_model(data_dir, num_epochs=100, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()
    print(f"Using device: {device}")
    
    train_loader, val_loader, classes = create_data_loaders(data_dir, batch_size=batch_size)
    print(f"Number of classes: {len(classes)}")
    print(f"Classes: {classes}")
    
    model = FoodClassifier(len(classes))
    model = model.to(device)
    
    # Use label smoothing and focal loss combination
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Use AdamW with a learning rate schedule
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.attention.parameters(), 'lr': 2e-4},
        {'params': model.texture_conv.parameters(), 'lr': 2e-4},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n and 'attention' not in n and 'texture_conv' not in n], 'lr': 1e-4}
    ], weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[1e-5, 2e-4, 2e-4, 1e-4],
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        anneal_strategy='cos'
    )
    
    scaler = GradScaler()
    
    best_f1 = 0.0
    patience = 15
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        model.train()
        running_loss = 0.0
        train_preds = []
        train_targets = []
        
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Apply mixup augmentation
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, use_cuda=use_cuda)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            # For metrics, use original labels
            train_preds.extend(predicted.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
        
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        val_logits = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_logits.extend(outputs.cpu().numpy())
                
                _, predicted = outputs.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            train_targets, train_preds, average='weighted', zero_division=0
        )
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_targets, val_preds, average='weighted', zero_division=0
        )
        
        
        print(f'Train Loss: {running_loss/len(train_loader):.4f}, Train F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val F1: {val_f1:.4f}')
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            print(f'Saving best model with F1 score: {best_f1:.4f}')
            
            # Find optimal temperature
            optimal_temperature = find_optimal_temperature(np.array(val_logits), np.array(val_targets))
            print(f'Optimal temperature: {optimal_temperature:.2f}')
            
            # Save the model with temperature
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'class_names': classes,
                'temperature': optimal_temperature
            }, 'models/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping after {patience} epochs without improvement')
                break

    
if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    train_model('data')

print("Training complete. The best model has been saved in the 'models' directory.")

